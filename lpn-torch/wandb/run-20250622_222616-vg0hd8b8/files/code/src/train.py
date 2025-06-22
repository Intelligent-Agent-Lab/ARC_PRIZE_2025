import json
import logging
import math
import os
from typing import Optional, Dict, Any, Tuple
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from matplotlib import pyplot as plt
import tqdm
from tqdm.auto import trange as tqdm_trange
import wandb
import hydra
import omegaconf
import numpy as np

from src.models.lpn import LPN
from src.evaluator import Evaluator
from src.models.transformer import EncoderTransformer, DecoderTransformer
from src.visualization import (
    visualize_dataset_generation,
    visualize_heatmap,
    visualize_tsne,
    visualize_json_submission,
)
from src.data_utils import (
    load_datasets,
    shuffle_dataset_into_batches,
    data_augmentation_fn,
    make_leave_one_out,
    DATASETS_BASE_PATH,
)
from src.datasets.task_gen.dataloader import make_task_gen_dataloader, make_dataset


logging.getLogger().setLevel(logging.INFO)


class Trainer:
    def __init__(self, cfg: omegaconf.DictConfig, model: LPN, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        
        self.batch_size = cfg.training.batch_size
        self.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
        
        if self.batch_size % self.gradient_accumulation_steps != 0:
            raise ValueError(
                f"Batch size {self.batch_size} is not divisible by gradient accumulation steps {self.gradient_accumulation_steps}."
            )
        
        self.prior_kl_coeff = cfg.training.get("prior_kl_coeff")
        self.pairwise_kl_coeff = cfg.training.get("pairwise_kl_coeff")
        self.train_inference_mode = cfg.training.inference_mode
        self.train_inference_kwargs = cfg.training.get("inference_kwargs") or {}
        
        # Handle training datasets
        if cfg.training.train_datasets and cfg.training.task_generator:
            raise ValueError("Only one of 'train_datasets' and 'task_generator' can be specified.")
        
        if cfg.training.train_datasets:
            self.task_generator = False
            train_datasets = cfg.training.train_datasets
            if isinstance(train_datasets, str):
                train_datasets = [train_datasets]
            try:
                train_dataset_grids, train_dataset_shapes = [], []
                for grids, shapes, _ in load_datasets(train_datasets, cfg.training.get("use_hf", True)):
                    assert grids.shape[0:1] == shapes.shape[0:1]
                    train_dataset_grids.append(grids)
                    train_dataset_shapes.append(shapes)
                self.train_dataset_grids = torch.cat(train_dataset_grids, dim=0)
                self.train_dataset_shapes = torch.cat(train_dataset_shapes, dim=0)
                self.init_grids = self.train_dataset_grids[:1]
                self.init_shapes = self.train_dataset_shapes[:1]
            except Exception as e:
                logging.error(f"Error loading training datasets: {e}")
                raise
            logging.info(f"Train dataset shape: {self.train_dataset_grids.shape}")
        
        if cfg.training.task_generator:
            logging.info("Using a task generator for training.")
            self.task_generator = True
            self.task_generator_kwargs = cfg.training.task_generator
            for arg in ["num_workers", "num_pairs", "class"]:
                assert arg in self.task_generator_kwargs, f"Task generator must have arg '{arg}'."
            num_pairs = self.task_generator_kwargs["num_pairs"]
            num_rows, num_cols = self.model.encoder.config.max_rows, self.model.encoder.config.max_cols
            self.init_grids = torch.zeros((1, num_pairs, num_rows, num_cols, 2), dtype=torch.uint8)
            self.init_shapes = torch.ones((1, num_pairs, 2, 2), dtype=torch.uint8)
        
        self.online_data_augmentation = cfg.training.online_data_augmentation
        
        # Load eval datasets
        self.eval_datasets = []
        for dict_ in cfg.eval.eval_datasets or []:
            for arg in ["folder"]:
                assert arg in dict_, f"Each eval dataset must have arg '{arg}'."
            folder, length, seed = dict_["folder"], dict_.get("length"), dict_.get("seed", 0)
            grids, shapes, _ = load_datasets([folder], dict_.get("use_hf", True))[0]
            if length is not None:
                torch.manual_seed(seed)
                indices = torch.randperm(len(grids))[:length]
                grids, shapes = grids[indices], shapes[indices]
            batch_size = dict_.get("batch_size", len(grids))
            # Drop the last batch if it's not full
            num_batches = len(grids) // batch_size
            grids, shapes = grids[:num_batches * batch_size], shapes[:num_batches * batch_size]
            self.eval_datasets.append({
                "dataset_name": folder.rstrip().split("/")[-1],
                "dataset_grids": grids,
                "dataset_shapes": shapes,
                "batch_size": batch_size,
            })
        
        # Load test datasets
        self.test_datasets = []
        for i, dict_ in enumerate(cfg.eval.test_datasets or []):
            if dict_.get("generator", False):
                for arg in ["num_pairs", "length"]:
                    assert arg in dict_, f"Each test generator dataset must have arg '{arg}'."
                num_pairs, length = dict_["num_pairs"], dict_["length"]
                default_dataset_name = "generator"
                task_generator_kwargs = dict_.get("task_generator_kwargs") or {}
                grids, shapes, program_ids = make_dataset(
                    length,
                    num_pairs,
                    num_workers=8,
                    task_generator_class=dict_["generator"],
                    online_data_augmentation=self.online_data_augmentation,
                    seed=dict_.get("seed", 0),
                    **task_generator_kwargs,
                )
            else:
                for arg in ["folder", "length"]:
                    assert arg in dict_, f"Each test dataset must have arg '{arg}'."
                folder, length = dict_["folder"], dict_["length"]
                default_dataset_name = folder.rstrip().split("/")[-1]
                grids, shapes, program_ids = load_datasets([folder], dict_.get("use_hf", True))[0]
            
            if length is not None:
                torch.manual_seed(dict_.get("seed", 0))
                indices = torch.randperm(len(grids))[:length]
                # Ensure all are torch.Tensor for torch indexing
            def to_tensor(x):
                import torch
                import numpy as np
                # JAX array: to numpy first
                if hasattr(x, 'to_py'):
                    arr = x.to_py()
                    if hasattr(arr, 'shape') and len(arr.shape) == 0:
                        return torch.tensor([])  # empty tensor
                    return torch.from_numpy(arr)
                # numpy array
                elif isinstance(x, np.ndarray):
                    if x.shape == ():  # scalar/empty
                        return torch.tensor([])
                    return torch.from_numpy(x)
                # torch tensor
                elif isinstance(x, torch.Tensor):
                    return x
                # list or other
                elif isinstance(x, (list, tuple)):
                    if len(x) == 0:
                        return torch.tensor([])
                    return torch.tensor(x)
                else:
                    return torch.tensor(x)
                grids = to_tensor(grids)
                shapes = to_tensor(shapes)
                program_ids = to_tensor(program_ids)
                # indices는 이미 torch.randperm로 생성됨
                grids, shapes, program_ids = grids[indices], shapes[indices], program_ids[indices]
            
            batch_size = dict_.get("batch_size", len(grids))
            # Drop the last batch if it's not full
            num_batches = len(grids) // batch_size
            grids, shapes, program_ids = (
                grids[:num_batches * batch_size],
                shapes[:num_batches * batch_size],
                program_ids[:num_batches * batch_size],
            )
            
            inference_mode = dict_.get("inference_mode", "mean")
            test_name = default_dataset_name + "_" + dict_.get("name", f"{inference_mode}_{i}")
            inference_kwargs = dict_.get("inference_kwargs", {})
            
            self.test_datasets.append({
                "test_name": test_name,
                "dataset_grids": grids,
                "dataset_shapes": shapes,
                "batch_size": batch_size,
                "num_tasks_to_show": dict_.get("num_tasks_to_show", 5),
                "program_ids": program_ids,
                "inference_mode": inference_mode,
                "inference_kwargs": inference_kwargs,
            })
        
        # Load json datasets
        self.json_datasets = []
        for i, dict_ in enumerate(cfg.eval.json_datasets or []):
            for arg in ["challenges", "solutions"]:
                assert arg in dict_, f"Each json dataset must have arg '{arg}'."
            json_challenges_file = dict_["challenges"]
            json_solutions_file = dict_["solutions"]
            inference_mode = dict_.get("inference_mode", "mean")
            default_dataset_name = json_challenges_file.rstrip().split("/")[-1].split(".")[0]
            test_name = default_dataset_name + "_" + dict_.get("name", f"{inference_mode}_{i}")
            evaluator = Evaluator(
                self.model,
                inference_mode=inference_mode,
                inference_mode_kwargs=dict_.get("inference_kwargs", {}),
                device=self.device,
            )
            self.json_datasets.append({
                "test_name": test_name,
                "json_challenges_file": os.path.join(DATASETS_BASE_PATH, json_challenges_file),
                "json_solutions_file": os.path.join(DATASETS_BASE_PATH, json_solutions_file),
                "evaluator": evaluator,
                "only_n_tasks": dict_.get("only_n_tasks", None),
                "num_tasks_to_show": dict_.get("num_tasks_to_show", 5),
                "overfit_task": dict_.get("overfit_task", None),
            })

    def init_train_state(
        self, learning_rate: float, linear_warmup_steps: int = 99
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR]:
        """Initialize optimizer and scheduler."""
        
        # Create optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Create learning rate scheduler with warmup
        def lr_lambda(step):
            if step < linear_warmup_steps:
                return step / linear_warmup_steps
            return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def train_one_step(
        self, 
        optimizer: optim.Optimizer,
        batch: Tuple[torch.Tensor, torch.Tensor],
        accumulate_gradients: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Train one step."""
        pairs, grid_shapes = batch
        pairs = pairs.to(self.device)
        grid_shapes = grid_shapes.to(self.device)
        
        # Clear gradients if not accumulating
        if not accumulate_gradients:
            optimizer.zero_grad()
        
        # Forward pass
        loss, metrics = self.model(
            pairs,
            grid_shapes,
            training=True,
            prior_kl_coeff=self.prior_kl_coeff,
            pairwise_kl_coeff=self.pairwise_kl_coeff,
            mode=self.train_inference_mode,
            **self.train_inference_kwargs,
        )
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** (1. / 2)
        
        metrics["grad_norm"] = torch.tensor(grad_norm)
        return metrics

    def train_n_steps(
        self, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LambdaLR,
        batches: Tuple[torch.Tensor, torch.Tensor],
        num_steps: int
    ) -> Dict[str, torch.Tensor]:
        """Train multiple steps with gradient accumulation."""
        
        all_metrics = []
        
        for step in range(num_steps):
            # Get batch for this step
            batch_pairs = batches[0][step]
            batch_shapes = batches[1][step]
            batch = (batch_pairs, batch_shapes)
            
            # Determine if this is the last accumulation step
            is_accumulation_step = (step + 1) % self.gradient_accumulation_steps == 0
            
            # Train step
            metrics = self.train_one_step(
                optimizer, 
                batch, 
                accumulate_gradients=not is_accumulation_step
            )
            all_metrics.append(metrics)
            
            # Update weights if accumulation is complete
            if is_accumulation_step:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = torch.stack([m[key] for m in all_metrics]).mean()
        
        return avg_metrics

    def prepare_train_dataset_for_epoch(
        self, log_every_n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shuffle and prepare training dataset for epoch."""
        
        # Shuffle dataset
        indices = torch.randperm(self.train_dataset_grids.shape[0])
        grids = self.train_dataset_grids[indices]
        shapes = self.train_dataset_shapes[indices]
        
        # Reshape into batches
        grids, shapes = shuffle_dataset_into_batches(grids, shapes, self.batch_size, None)
        
        # Trim to fit log_every_n_steps
        num_logs = grids.shape[0] // log_every_n_steps
        grids = grids[:num_logs * log_every_n_steps]
        shapes = shapes[:num_logs * log_every_n_steps]
        
        # Apply data augmentation if enabled
        if self.online_data_augmentation:
            grids, shapes = data_augmentation_fn(grids, shapes, None)
        
        # Reshape for logging intervals
        grids = grids.view(num_logs, log_every_n_steps, *grids.shape[1:])
        shapes = shapes.view(num_logs, log_every_n_steps, *shapes.shape[1:])
        
        return grids, shapes

    def eval(
        self,
        dataset_name: str,
        dataset_grids: torch.Tensor,
        dataset_shapes: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, float]:
        """Evaluate the model on given dataset."""
        
        self.model.eval()
        all_metrics = []
        
        # Split into batches
        num_batches = len(dataset_grids) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_grids = dataset_grids[start_idx:end_idx].to(self.device)
                batch_shapes = dataset_shapes[start_idx:end_idx].to(self.device)
                
                _, metrics = self.model(
                    batch_grids,
                    batch_shapes,
                    training=False,
                    prior_kl_coeff=self.prior_kl_coeff,
                    pairwise_kl_coeff=self.pairwise_kl_coeff,
                    mode=self.train_inference_mode,
                    **self.train_inference_kwargs,
                )
                all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key].item() if torch.is_tensor(m[key]) else m[key] for m in all_metrics]
            avg_metrics[f"eval/{dataset_name}/{key}"] = np.mean(values)
        
        self.model.train()
        return avg_metrics

    def test_dataset_submission(
        self,
        test_name: str,
        dataset_grids: torch.Tensor,
        dataset_shapes: torch.Tensor,
        program_ids: Optional[torch.Tensor],
        batch_size: int,
        num_tasks_to_show: int = 5,
        inference_mode: str = "mean",
        inference_kwargs: Dict = None,
    ) -> Tuple[Dict[str, float], Optional[plt.Figure], plt.Figure, Optional[plt.Figure]]:
        """Test model on dataset by generating outputs."""
        
        if inference_kwargs is None:
            inference_kwargs = {}
        
        self.model.eval()
        
        # Create leave-one-out datasets
        leave_one_out_grids = make_leave_one_out(dataset_grids, axis=-4)
        leave_one_out_shapes = make_leave_one_out(dataset_shapes, axis=-3)
        
        all_generated_grids = []
        all_generated_shapes = []
        all_contexts = []
        all_metrics = []
        
        num_batches = len(dataset_grids) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_leave_one_out_grids = leave_one_out_grids[start_idx:end_idx].to(self.device)
                batch_leave_one_out_shapes = leave_one_out_shapes[start_idx:end_idx].to(self.device)
                batch_grids = dataset_grids[start_idx:end_idx].to(self.device)
                batch_shapes = dataset_shapes[start_idx:end_idx].to(self.device)
                
                grids_inputs = batch_grids[..., 0]
                grids_outputs = batch_grids[..., 1]
                shapes_inputs = batch_shapes[..., 0]
                shapes_outputs = batch_shapes[..., 1]
                
                # Generate outputs
                generated_grids_outputs, generated_shapes_outputs, info = self.model.generate_output(
                    batch_leave_one_out_grids,
                    batch_leave_one_out_shapes,
                    grids_inputs,
                    shapes_inputs,
                    training=False,
                    mode=inference_mode,
                    **inference_kwargs
                )
                
                # Compute metrics
                correct_shapes = torch.all(generated_shapes_outputs == shapes_outputs, dim=-1)
                
                # Create masks for valid pixels
                max_rows, max_cols = self.model.decoder.config.max_rows, self.model.decoder.config.max_cols
                row_arange = torch.arange(max_rows, device=self.device).view(1, 1, max_rows)
                col_arange = torch.arange(max_cols, device=self.device).view(1, 1, max_cols)
                
                input_row_mask = row_arange < shapes_outputs[..., :1]
                input_col_mask = col_arange < shapes_outputs[..., 1:]
                input_mask = input_row_mask.unsqueeze(-1) & input_col_mask.unsqueeze(-2)
                
                pixels_equal = torch.where(
                    input_mask & correct_shapes.unsqueeze(-1).unsqueeze(-1),
                    (generated_grids_outputs == grids_outputs),
                    torch.tensor(False, device=self.device)
                )
                
                pixel_correctness = pixels_equal.sum(dim=(-1, -2)).float() / shapes_outputs.prod(dim=-1).float()
                accuracy = (pixels_equal.sum(dim=(-1, -2)) == shapes_outputs.prod(dim=-1)).float()
                
                batch_metrics = {
                    "correct_shapes": correct_shapes.float().mean(),
                    "pixel_correctness": pixel_correctness.mean(),
                    "accuracy": accuracy.mean(),
                }
                
                # Store results
                all_generated_grids.append(generated_grids_outputs.cpu())
                all_generated_shapes.append(generated_shapes_outputs.cpu())
                all_contexts.append(info["context"].cpu())
                all_metrics.append(batch_metrics)
        
        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key].item() if torch.is_tensor(m[key]) else m[key] for m in all_metrics]
            avg_metrics[f"test/{test_name}/{key}"] = np.mean(values)
        
        # Concatenate results
        generated_grids = torch.cat(all_generated_grids, dim=0)
        generated_shapes = torch.cat(all_generated_shapes, dim=0)
        contexts = torch.cat(all_contexts, dim=0)
        
        # Create visualizations
        if num_tasks_to_show > 0:
            fig_grids = visualize_dataset_generation(
                dataset_grids[:num_tasks_to_show], 
                dataset_shapes[:num_tasks_to_show], 
                generated_grids[:num_tasks_to_show], 
                generated_shapes[:num_tasks_to_show], 
                num_tasks_to_show
            )
        else:
            fig_grids = None
        
        # Create heatmap
        max_rows, max_cols = self.model.decoder.config.max_rows, self.model.decoder.config.max_cols
        grid_row_mask = torch.arange(max_rows).unsqueeze(-1) < dataset_shapes[..., 0, 1:]
        grid_col_mask = torch.arange(max_cols).unsqueeze(0) < dataset_shapes[..., 1, 1:]
        grid_pad_mask = grid_row_mask.unsqueeze(-1) & grid_col_mask.unsqueeze(-2)
        
        pixel_correct_binary = (generated_grids == dataset_grids[..., 1]) * grid_pad_mask
        pixel_accuracy = pixel_correct_binary.sum(dim=(0, 1)) / (grid_pad_mask.sum(dim=(0, 1)) + 1e-5)
        pixel_frequency = grid_pad_mask.sum(dim=(0, 1)) / grid_pad_mask.sum()
        
        fig_heatmap = visualize_heatmap(pixel_accuracy, pixel_frequency)
        
        # Create t-SNE plot
        if program_ids is not None:
            pairs_per_problem = contexts.shape[-2]
            program_ids_expanded = program_ids.repeat_interleave(pairs_per_problem)
            contexts_flat = contexts.view(-1, contexts.shape[-1])
            fig_latents = visualize_tsne(contexts_flat, program_ids_expanded)
        else:
            fig_latents = None
        
        self.model.train()
        return avg_metrics, fig_grids, fig_heatmap, fig_latents

    @classmethod
    def test_json_submission(
        cls,
        model: LPN,
        evaluator: Evaluator,
        json_challenges_file: str,
        json_solutions_file: str,
        test_name: str,
        only_n_tasks: Optional[int] = None,
        overfit_task: Optional[str] = None,
        num_tasks_to_show: int = 5,
        progress_bar: bool = False,
    ) -> Tuple[Dict[str, float], Optional[plt.Figure]]:
        """Test model on JSON submission format."""
        
        with open(json_challenges_file, "r") as f:
            challenges = json.load(f)
        
        train = "training" in json_challenges_file
        generations = evaluator.json_submission(
            challenges, only_n_tasks, overfit_task, progress_bar, train=train
        )
        
        with open(json_solutions_file, "r") as f:
            solutions = json.load(f)
        
        metrics = evaluator.evaluate_generations(generations, solutions)
        metrics = {f"test/{test_name}/{k}": v for k, v in metrics.items()}
        
        if num_tasks_to_show > 0:
            fig_grids = visualize_json_submission(challenges, generations, solutions, num_tasks_to_show)
        else:
            fig_grids = None
        
        return metrics, fig_grids

    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LambdaLR,
        trange: tqdm.tqdm,
        total_num_steps: int,
        log_every_n_steps: int,
        eval_every_n_logs: Optional[int] = None,
        save_checkpoint_every_n_logs: Optional[int] = None,
    ) -> None:
        """Train for one epoch."""
        
        if self.task_generator:
            # Use task generator
            task_generator_kwargs = dict(self.task_generator_kwargs)
            num_workers = task_generator_kwargs.pop("num_workers")
            task_generator_class = task_generator_kwargs.pop("class")
            num_pairs = task_generator_kwargs.pop("num_pairs")
            dataloader = make_task_gen_dataloader(
                batch_size=self.batch_size,
                log_every_n_steps=log_every_n_steps,
                num_workers=num_workers,
                task_generator_class=task_generator_class,
                num_pairs=num_pairs,
                online_data_augmentation=self.online_data_augmentation,
                **task_generator_kwargs,
            )
        else:
            # Use prepared dataset
            grids, shapes = self.prepare_train_dataset_for_epoch(log_every_n_steps)
            dataloader = zip(grids, shapes)
        
        dataloading_time = time.time()
        for batches in dataloader:
            wandb.log({"timing/dataloading_time": time.time() - dataloading_time})
            
            # Training
            start = time.time()
            metrics = self.train_n_steps(optimizer, scheduler, batches, log_every_n_steps)
            end = time.time()
            
            trange.update(log_every_n_steps)
            self.num_steps += log_every_n_steps
            self.num_logs += 1
            
            throughput = log_every_n_steps * self.batch_size / (end - start)
            metrics.update({
                "timing/train_time": end - start,
                "timing/train_num_samples_per_second": throughput
            })
            
            # Convert tensor metrics to float for wandb
            wandb_metrics = {}
            for k, v in metrics.items():
                if torch.is_tensor(v):
                    wandb_metrics[k] = v.item()
                else:
                    wandb_metrics[k] = v
            
            wandb.log(wandb_metrics, step=self.num_steps)
            
            # Save checkpoint
            if save_checkpoint_every_n_logs and self.num_logs % save_checkpoint_every_n_logs == 0:
                self.save_checkpoint("model.pt")
            
            # Evaluation
            if eval_every_n_logs and self.num_logs % eval_every_n_logs == 0:
                # Eval datasets
                for dataset_dict in self.eval_datasets:
                    start = time.time()
                    eval_metrics = self.eval(**dataset_dict)
                    eval_metrics[f"timing/eval_{dataset_dict['dataset_name']}"] = time.time() - start
                    wandb.log(eval_metrics, step=self.num_steps)
                
                # Dataset test
                for dataset_dict in self.test_datasets:
                    start = time.time()
                    test_metrics, fig_grids, fig_heatmap, fig_latents = self.test_dataset_submission(**dataset_dict)
                    test_metrics[f"timing/test_{dataset_dict['test_name']}"] = time.time() - start
                    
                    # Add figures to metrics
                    for fig, name in [
                        (fig_grids, "generation"),
                        (fig_heatmap, "pixel_accuracy"),
                        (fig_latents, "latents"),
                    ]:
                        if fig is not None:
                            test_metrics[f"test/{dataset_dict['test_name']}/{name}"] = wandb.Image(fig)
                    
                    wandb.log(test_metrics, step=self.num_steps)
                    plt.close('all')
                
                # Json test
                for json_file_dict in self.json_datasets:
                    start = time.time()
                    test_metrics, fig_grids = self.test_json_submission(**json_file_dict)
                    json_test_name = json_file_dict["test_name"]
                    test_metrics[f"timing/test_{json_test_name}"] = time.time() - start
                    if fig_grids is not None:
                        test_metrics[f"test/{json_test_name}/generation"] = wandb.Image(fig_grids)
                    wandb.log(test_metrics, step=self.num_steps)
            
            # Exit if the total number of steps is reached
            if self.num_steps >= total_num_steps:
                break
            
            dataloading_time = time.time()

    def train(
        self,
        cfg: omegaconf.DictConfig,
        progress_bar: bool = True,
        start_num_steps: int = 0,
    ) -> None:
        """Main training loop."""
        
        # Initialize optimizer and scheduler
        optimizer, scheduler = self.init_train_state(cfg.training.learning_rate)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        num_params_encoder = sum(p.numel() for p in self.model.encoder.parameters())
        num_params_decoder = sum(p.numel() for p in self.model.decoder.parameters())
        
        total_num_steps: int = cfg.training.total_num_steps
        log_every_n_steps: int = cfg.training.log_every_n_steps
        eval_every_n_logs: Optional[int] = cfg.training.eval_every_n_logs
        save_checkpoint_every_n_logs: Optional[int] = cfg.training.save_checkpoint_every_n_logs
        
        self.num_steps, self.num_logs = start_num_steps, 0
        
        logging.info("Starting training...")
        logging.info(f"Number of total parameters: {num_params:,}")
        logging.info(f"Number of encoder parameters: {num_params_encoder:,}")
        logging.info(f"Number of decoder parameters: {num_params_decoder:,}")
        logging.info(f"Using device: {self.device}")
        logging.info(f"Total number of gradient steps: {total_num_steps:,}")
        
        if not self.task_generator:
            num_logs_per_epoch = self.train_dataset_grids.shape[0] // (log_every_n_steps * self.batch_size)
            if num_logs_per_epoch == 0:
                raise ValueError(
                    "The number of logs per epoch is 0 because the dataset size is "
                    f"{self.train_dataset_grids.shape[0]} < {self.batch_size=} * {log_every_n_steps=}."
                )
            num_steps_per_epoch = num_logs_per_epoch * log_every_n_steps
            num_epochs = math.ceil(total_num_steps / num_steps_per_epoch)
            
            logging.info(f"Number of epochs: {num_epochs:,}")
            logging.info(f"Number of logs per epoch: {num_logs_per_epoch:,}")
            logging.info(f"Number of gradient steps per epoch: {num_steps_per_epoch:,}")
            logging.info(f"Total number of logs: {num_logs_per_epoch * num_epochs:,}")
        else:
            num_epochs = 1
            logging.info(f"Total number of logs: {total_num_steps // log_every_n_steps:,}")
        
        logging.info(f"Logging every {log_every_n_steps:,} gradient steps.")
        
        if eval_every_n_logs:
            steps_between_evals = eval_every_n_logs * log_every_n_steps
            logging.info(f"Total number of evaluations: {total_num_steps // steps_between_evals:,}")
            logging.info(f"Evaluating every {steps_between_evals:,} gradient steps.")
        else:
            logging.info("Not evaluating during training.")
        
        if save_checkpoint_every_n_logs:
            steps_between_checkpoints = save_checkpoint_every_n_logs * log_every_n_steps
            logging.info(f"Total number of checkpoints: {total_num_steps // steps_between_checkpoints:,}")
            logging.info(f"Saving a checkpoint every {steps_between_checkpoints:,} gradient steps.")
        else:
            logging.info("Not saving checkpoints during training.")
        
        trange = tqdm_trange(total_num_steps, disable=not progress_bar)
        
        try:
            for _ in range(num_epochs):
                self.train_epoch(
                    optimizer,
                    scheduler,
                    trange,
                    total_num_steps,
                    log_every_n_steps,
                    eval_every_n_logs,
                    save_checkpoint_every_n_logs,
                )
        except KeyboardInterrupt:
            logging.info("Interrupted training.")

    def save_checkpoint(self, ckpt_path: str) -> None:
        """Save model checkpoint."""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_steps': self.num_steps,
        }
        
        torch.save(checkpoint, ckpt_path)
        
        # Save to wandb
        run_name = self.make_safe_run_name(wandb.run.name)
        artifact = wandb.Artifact(f"{run_name}--checkpoint", type="model", metadata=dict(wandb.run.config))
        artifact.add_file(ckpt_path)
        wandb.run.log_artifact(artifact, name="checkpoint", aliases=["latest", f"num_steps_{self.num_steps}"])

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, model: LPN) -> int:
        """Load model checkpoint and return number of steps."""
        
        if checkpoint_path.startswith("wandb://"):
            # Load from wandb artifact
            artifact = wandb.use_artifact(checkpoint_path)
            artifact_dir = artifact.download()
            checkpoint_file = os.path.join(artifact_dir, "model.pt")
            if not os.path.exists(checkpoint_file):
                checkpoint_file = os.path.join(artifact_dir, "state.msgpack")  # Fallback for JAX checkpoints
        else:
            checkpoint_file = checkpoint_path
        
        if checkpoint_file.endswith('.pt') or checkpoint_file.endswith('.pth'):
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            start_num_steps = checkpoint.get('num_steps', 0)
        else:
            # Handle other formats or convert from JAX
            logging.warning(f"Checkpoint format not directly supported: {checkpoint_file}")
            start_num_steps = 0
        
        return start_num_steps

    @classmethod
    def make_safe_run_name(cls, run_name: str) -> str:
        """Make wandb run name safe for file systems."""
        return (
            run_name.replace(",", ".")
            .replace(":", "")
            .replace(" ", "")
            .replace("(", "_")
            .replace(")", "_")
            .replace("[", "_")
            .replace("]", "_")
            .replace("+", "_")
            .replace("=", "_")
        )


def instantiate_config_for_mpt(
    transformer_cfg: omegaconf.DictConfig,
) -> Any:
    """Override the TransformerLayer config for mixed-precision training."""
    config = hydra.utils.instantiate(
        transformer_cfg,
        transformer_layer=hydra.utils.instantiate(
            transformer_cfg.transformer_layer, 
            dtype=torch.bfloat16
        ),
    )
    return config


@hydra.main(config_path="configs", version_base=None, config_name="task_gen")
def run(cfg: omegaconf.DictConfig):
    """Main training script."""
    
    logging.info(f"Available devices: {torch.cuda.device_count() if torch.cuda.is_available() else 'CPU only'}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
    
    # Instantiate model
    if cfg.training.get("mixed_precision", False):
        encoder = EncoderTransformer(instantiate_config_for_mpt(cfg.encoder_transformer))
        decoder = DecoderTransformer(instantiate_config_for_mpt(cfg.decoder_transformer))
    else:
        encoder = EncoderTransformer(hydra.utils.instantiate(cfg.encoder_transformer))
        decoder = DecoderTransformer(hydra.utils.instantiate(cfg.decoder_transformer))
    
    lpn = LPN(encoder=encoder, decoder=decoder)
    
    # Initialize wandb
    wandb.init(
        entity="kyushikmin",
        project="ARC-lpn",
        settings=wandb.Settings(console="redirect"),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        save_code=True,
    )
    
    # Initialize trainer
    trainer = Trainer(cfg=cfg, model=lpn, device=device)
    
    # Load checkpoint if resuming
    start_num_steps = 0
    if cfg.training.get("resume_from_checkpoint", False):
        checkpoint_path = cfg.training.resume_from_checkpoint
        logging.info(f"Resuming from checkpoint: {checkpoint_path}...")
        start_num_steps = trainer.load_checkpoint(checkpoint_path, lpn)
    
    # Start training
    trainer.train(
        cfg=cfg,
        progress_bar=True,
        start_num_steps=start_num_steps,
    )
    
    # Save final checkpoint
    trainer.save_checkpoint("final_model.pt")


if __name__ == "__main__":
    run()