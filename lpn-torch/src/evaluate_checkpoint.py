"""
Example usages:

python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/upbeat-wildflower-739--checkpoint:v9 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i mean \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/upbeat-wildflower-739--checkpoint:v9 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 20 \
    --lr 5e-2 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/playful-monkey-758--checkpoint:v1 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i random_search \
    --num-samples 100 \
    --scale 1.0 \
    --scan-batch-size 10 \
    --random-search-seed 0
"""

import argparse
import os
from typing import Optional, Dict
import pickle

import wandb
import hydra
import omegaconf
import torch
import torch.nn as nn
import json
from tqdm import trange

# Import your PyTorch modules (replace with actual imports)
# from src.models.lpn import LPN
# from src.evaluator import Evaluator
# from src.models.transformer import EncoderTransformer, DecoderTransformer
# from src.train import Trainer, load_datasets, instantiate_config_for_mpt
# from src.data_utils import make_leave_one_out, DATASETS_BASE_PATH

# Mock imports for demonstration
class LPN(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

class EncoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

class DecoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

class Evaluator:
    def __init__(self, model, inference_mode, inference_mode_kwargs, device=None):
        self.model = model
        self.inference_mode = inference_mode
        self.inference_mode_kwargs = inference_mode_kwargs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    @staticmethod
    def test_json_submission(*args, **kwargs):
        return {}, None

def load_datasets(datasets, use_hf=True):
    # Mock implementation
    return [(torch.randn(100, 3, 30, 30, 2), torch.randint(1, 31, (100, 3, 2, 2)), None)]

def make_leave_one_out(x, axis):
    # Mock implementation
    return x

DATASETS_BASE_PATH = "."

def instantiate_config_for_mpt(config):
    return hydra.utils.instantiate(config)


def instantiate_model(cfg: omegaconf.DictConfig, mixed_precision: bool, device: torch.device) -> LPN:
    """Instantiate the LPN model from configuration."""
    if mixed_precision:
        encoder = EncoderTransformer(instantiate_config_for_mpt(cfg.encoder_transformer))
        decoder = DecoderTransformer(instantiate_config_for_mpt(cfg.decoder_transformer))
    else:
        encoder = EncoderTransformer(hydra.utils.instantiate(cfg.encoder_transformer))
        decoder = DecoderTransformer(hydra.utils.instantiate(cfg.decoder_transformer))
    
    lpn = LPN(encoder=encoder, decoder=decoder)
    lpn.to(device)
    lpn.eval()  # Set to evaluation mode
    return lpn


def load_model_weights(model: nn.Module, artifact_dir: str, ckpt_name: str = "state.pt") -> nn.Module:
    """Load model weights from checkpoint file."""
    checkpoint_path = os.path.join(artifact_dir, ckpt_name)
    
    # Try different possible checkpoint formats
    possible_files = [ckpt_name, "state.pt", "model.pt", "checkpoint.pt", "state.msgpack"]
    
    for filename in possible_files:
        filepath = os.path.join(artifact_dir, filename)
        if os.path.exists(filepath):
            try:
                if filename.endswith('.pt') or filename.endswith('.pth'):
                    # PyTorch checkpoint
                    checkpoint = torch.load(filepath, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        elif 'params' in checkpoint:
                            model.load_state_dict(checkpoint['params'])
                        else:
                            model.load_state_dict(checkpoint)
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"Loaded checkpoint from {filepath}")
                    return model
                elif filename.endswith('.msgpack'):
                    # For msgpack files (JAX format), you might need conversion
                    print(f"Warning: msgpack file found ({filepath}). Consider converting to PyTorch format.")
                    continue
                elif filename.endswith('.pkl'):
                    # Pickle format
                    with open(filepath, 'rb') as f:
                        checkpoint = pickle.load(f)
                    model.load_state_dict(checkpoint)
                    print(f"Loaded checkpoint from {filepath}")
                    return model
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
                continue
    
    raise FileNotFoundError(f"No valid checkpoint found in {artifact_dir}")


def build_generate_output_batch(
    model: LPN, eval_inference_mode: str, eval_inference_mode_kwargs: dict, device: torch.device
) -> callable:
    """Build a function to generate output batches."""
    def generate_output_batch(
        leave_one_out_grids: torch.Tensor, 
        leave_one_out_shapes: torch.Tensor, 
        dataset_grids: torch.Tensor, 
        dataset_shapes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        with torch.no_grad():
            grids_inputs = dataset_grids[..., 0]
            labels_grids_outputs = dataset_grids[..., 1]
            shapes_inputs = dataset_shapes[..., 0]
            labels_shapes_outputs = dataset_shapes[..., 1]
            
            # Move tensors to device
            leave_one_out_grids = leave_one_out_grids.to(device)
            leave_one_out_shapes = leave_one_out_shapes.to(device)
            grids_inputs = grids_inputs.to(device)
            shapes_inputs = shapes_inputs.to(device)
            
            generated_grids_outputs, generated_shapes_outputs, _ = model.generate_output(
                leave_one_out_grids,
                leave_one_out_shapes,
                grids_inputs,
                shapes_inputs,
                training=False,
                mode=eval_inference_mode,
                **eval_inference_mode_kwargs
            )
            
            # Move back to CPU for evaluation
            generated_grids_outputs = generated_grids_outputs.cpu()
            generated_shapes_outputs = generated_shapes_outputs.cpu()
            
            correct_shapes = torch.all(generated_shapes_outputs == labels_shapes_outputs, dim=-1)
            
            # Create masks for valid pixels
            batch_shape = grids_inputs.shape[:-2]
            row_arange = torch.arange(grids_inputs.shape[-2]).view(
                *([1] * len(batch_shape)), grids_inputs.shape[-2]
            )
            col_arange = torch.arange(grids_inputs.shape[-1]).view(
                *([1] * len(batch_shape)), grids_inputs.shape[-1]
            )
            
            input_row_mask = row_arange < labels_shapes_outputs[..., :1]
            input_col_mask = col_arange < labels_shapes_outputs[..., 1:]
            input_mask = input_row_mask.unsqueeze(-1) & input_col_mask.unsqueeze(-2)
            
            pixels_equal = torch.where(
                input_mask & correct_shapes.unsqueeze(-1).unsqueeze(-1),
                (generated_grids_outputs == labels_grids_outputs),
                torch.tensor(False)
            )
            
            pixel_correctness = pixels_equal.sum(dim=(-1, -2)).float() / labels_shapes_outputs.prod(dim=-1).float()
            accuracy = (pixels_equal.sum(dim=(-1, -2)) == labels_shapes_outputs.prod(dim=-1)).float()
            
            metrics = {
                "correct_shapes": correct_shapes.float().mean(),
                "pixel_correctness": pixel_correctness.mean(),
                "accuracy": accuracy.mean(),
            }
            return metrics
    
    return generate_output_batch


def evaluate_json(
    model: LPN,
    evaluator: Evaluator,
    json_challenges_file: str,
    json_solutions_file: str,
    only_n_tasks: Optional[int],
    random_search_seed: int,
) -> dict:
    """Evaluate model on JSON dataset."""
    print(f"Evaluating the model on {json_challenges_file.rstrip().split('/')[-1]}...")
    
    # Set random seed
    torch.manual_seed(random_search_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_search_seed)
    
    metrics, fig = Trainer.test_json_submission(
        model,
        evaluator,
        json_challenges_file=os.path.join(DATASETS_BASE_PATH, json_challenges_file),
        json_solutions_file=os.path.join(DATASETS_BASE_PATH, json_solutions_file),
        test_name="",
        only_n_tasks=only_n_tasks,
        progress_bar=True,
        num_tasks_to_show=0,
    )
    
    metrics = {k.split("/")[-1]: v for k, v in metrics.items()}
    metrics["fig"] = fig
    return metrics


def evaluate_custom_dataset(
    model: LPN,
    evaluator: Evaluator,
    dataset_folder: str,
    dataset_length: Optional[int],
    dataset_batch_size: int,
    dataset_use_hf: bool,
    dataset_seed: int,
    random_search_seed: int,
    device: torch.device,
) -> dict:
    """Evaluate model on custom dataset."""
    print(f"Evaluating the model on the {dataset_folder.rstrip().split('/')[-1]} dataset...")
    
    # Set random seeds
    torch.manual_seed(random_search_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_search_seed)
    
    # Load data
    grids, shapes, _ = load_datasets([dataset_folder], use_hf=dataset_use_hf)[0]
    
    if dataset_length is not None:
        torch.manual_seed(dataset_seed)
        indices = torch.randperm(len(grids))[:dataset_length]
        grids, shapes = grids[indices], shapes[indices]
    
    # Drop the last batch if it's smaller than the batch size
    num_batches = grids.shape[0] // dataset_batch_size
    grids = grids[:num_batches * dataset_batch_size]
    shapes = shapes[:num_batches * dataset_batch_size]
    
    leave_one_out_grids = make_leave_one_out(grids, axis=-4)
    leave_one_out_shapes = make_leave_one_out(shapes, axis=-3)
    
    # Reshape for batch processing
    grids = grids.view(num_batches, dataset_batch_size, *grids.shape[1:])
    shapes = shapes.view(num_batches, dataset_batch_size, *shapes.shape[1:])
    leave_one_out_grids = leave_one_out_grids.view(num_batches, dataset_batch_size, *leave_one_out_grids.shape[1:])
    leave_one_out_shapes = leave_one_out_shapes.view(num_batches, dataset_batch_size, *leave_one_out_shapes.shape[1:])
    
    generate_output_batch = build_generate_output_batch(
        model=evaluator.model,
        eval_inference_mode=evaluator.inference_mode,
        eval_inference_mode_kwargs=evaluator.inference_mode_kwargs,
        device=device
    )
    
    metrics_list = []
    for i in trange(num_batches, desc="Generating solutions"):
        batch_metrics = generate_output_batch(
            leave_one_out_grids[i],
            leave_one_out_shapes[i],
            grids[i],
            shapes[i],
        )
        metrics_list.append(batch_metrics)
    
    # Aggregate the metrics over the batches
    metrics = {
        k: torch.stack([m[k] for m in metrics_list]).mean().item() 
        for k in metrics_list[0].keys()
    }
    return metrics


def pretty_print(metrics: dict) -> None:
    """Pretty print evaluation metrics."""
    print("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (torch.Tensor, float, int)):
            if isinstance(v, torch.Tensor):
                v = v.item() if v.numel() == 1 else v
            if isinstance(v, (float, int)):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: not a scalar")
        else:
            print(f"{k}: not a scalar")


def main(
    artifact_path: str,
    json_challenges_file: Optional[str],
    json_solutions_file: Optional[str],
    only_n_tasks: Optional[int],
    dataset_folder: Optional[str],
    dataset_length: Optional[int],
    dataset_batch_size: Optional[int],
    dataset_use_hf: bool,
    dataset_seed: int,
    inference_mode: str,
    inference_mode_kwargs: dict,
    random_search_seed: int,
    mixed_precision: bool,
) -> None:
    """Main evaluation function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Make sure the wandb mode is enabled
    os.environ["WANDB_MODE"] = "run"
    
    print("Downloading the model artifact...")
    # Download the artifact and save the config file
    run = wandb.init()
    artifact = run.use_artifact(artifact_path, type="model")
    run.finish()
    cfg = omegaconf.OmegaConf.create(artifact.metadata)
    artifact_dir = artifact.download()
    omegaconf.OmegaConf.save(config=cfg, f=os.path.join(artifact_dir, "config.yaml"))
    
    print("Instantiating the model...")
    lpn = instantiate_model(cfg, mixed_precision, device)
    evaluator = Evaluator(
        lpn,
        inference_mode=inference_mode,
        inference_mode_kwargs=inference_mode_kwargs,
        device=device,
    )
    
    # Load the model weights
    print("Loading the model weights...")
    lpn = load_model_weights(lpn, artifact_dir)
    
    # Evaluate the model
    print(f"Inference mode: {evaluator.inference_mode}")
    kwargs = {k: v for k, v in evaluator.inference_mode_kwargs.items() if v is not None}
    if kwargs:
        print(f"Inference mode kwargs: {kwargs}")
    
    if json_challenges_file and json_solutions_file:
        metrics = evaluate_json(
            lpn,
            evaluator,
            json_challenges_file,
            json_solutions_file,
            only_n_tasks,
            random_search_seed,
        )
        pretty_print(metrics)
    
    if dataset_folder:
        if dataset_batch_size is None:
            dataset_batch_size = dataset_length or 32
        
        metrics = evaluate_custom_dataset(
            lpn,
            evaluator,
            dataset_folder,
            dataset_length,
            dataset_batch_size,
            dataset_use_hf,
            dataset_seed,
            random_search_seed,
            device,
        )
        pretty_print(metrics)


def true_or_false_from_arg(arg: str) -> bool:
    """Convert string argument to boolean."""
    if arg.lower() == "true":
        return True
    if arg.lower() == "false":
        return False
    raise ValueError(f"Invalid boolean argument '{arg}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model checkpoint on either the ARC json datasets or custom datasets. "
            "Must provide arguments for -w, and, either -jc and -js, or -d."
        )
    )
    parser.add_argument(
        "-w",
        "--wandb-artifact-path",
        type=str,
        required=True,
        help="WandB path to the desired artifact. E.g. 'TheThinker/ARC/faithful-dawn-316--checkpoint:v76'.",
    )
    parser.add_argument(
        "-jc",
        "--json-challenges-file",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON file with the ARC challenges. E.g. 'json/arc-agi_training_challenges.json'.",
    )
    parser.add_argument(
        "-js",
        "--json-solutions-file",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON file with the ARC solutions. E.g. 'json/arc-agi_training_solutions.json'.",
    )
    parser.add_argument(
        "--only-n-tasks",
        type=int,
        required=False,
        default=None,
        help="Number of tasks to evaluate the model on. 'None' to run on all tasks.",
    )
    parser.add_argument(
        "-d",
        "--dataset-folder",
        type=str,
        required=False,
        default=None,
        help="Path to the folder with the custom dataset. E.g. 'storage/v0_main_fix_test'.",
    )
    parser.add_argument(
        "--dataset-length",
        type=int,
        required=False,
        default=None,
        help="Number of examples to evaluate the model on. 'None' to run on all examples.",
    )
    parser.add_argument(
        "--dataset-batch-size",
        type=int,
        required=False,
        default=None,
        help="Batch size for the custom dataset evaluation. 'None' to use the length of the dataset.",
    )
    parser.add_argument(
        "--dataset-use-hf",
        type=true_or_false_from_arg,
        required=False,
        default=True,
        help="Whether to use Hugging Face to load the datasets (otherwise loads locally).",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        required=False,
        default=0,
        help="Seed to sample a subset of the custom dataset for evaluation.",
    )
    parser.add_argument(
        "-i",
        "--inference-mode",
        type=str,
        required=False,
        default="mean",
        help="Inference mode to use, choose from ['mean', 'first', 'random_search', 'gradient_ascent'].",
    )
    parser.add_argument(
        "--random-search-seed",
        type=int,
        required=False,
        default=0,
        help="Seed for the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        default=None,
        help="Number of samples for the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=False,
        default=None,
        help="Scale for the random noise added during the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        required=False,
        default=None,
        help="Batch size for the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=None,
        help="Number of steps for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=None,
        help="Learning rate for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to use a cosine decay learning rate schedule for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--lr-schedule-exponent",
        type=float,
        required=False,
        default=None,
        help="Exponent for the cosine decay learning rate schedule for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default=None,
        help="Optimizer to use for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--optimizer-kwargs",
        type=json.loads,
        required=False,
        default=None,
        help="Optimizer kwargs for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--accumulate-gradients-decoder-pairs",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to accumulate gradients for the decoder pairs in the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--scan-gradients-latents",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to scan gradients for the latents in the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--include-mean-latent",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to include the mean latent in the 'random_search' or 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--include-all-latents",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to include all latents in the 'random_search' or 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--random-perturbation",
        type=json.loads,
        required=False,
        default=None,
        help="Random perturbation kwargs. Requires 'num_samples' and 'scale' keys.",
    )
    parser.add_argument(
        "--mixed-precision",
        type=true_or_false_from_arg,
        required=False,
        default=True,
        help="Whether to use mixed precision for inference.",
    )
    
    args = parser.parse_args()
    
    # Validation
    if (
        args.json_challenges_file is None
        and args.json_solutions_file is not None
        or args.json_challenges_file is not None
        and args.json_solutions_file is None
    ):
        parser.error("Must provide both the json challenges (-jc) and solutions (-js) files.")
    
    if args.json_challenges_file is None and args.dataset_folder is None:
        parser.error(
            "Must provide either the json challenges (-jc) and solutions (-js) files or the dataset folder (-d)."
        )
    
    if args.inference_mode not in ["mean", "first", "random_search", "gradient_ascent"]:
        parser.error(
            "Invalid inference mode. Choose from ['mean', 'first', 'random_search', 'gradient_ascent']."
        )
    
    if args.inference_mode == "random_search":
        if args.num_samples is None:
            parser.error("The 'random_search' inference mode requires the --num-samples argument.")
        if args.scale is None:
            parser.error("The 'random_search' inference mode requires the --scale argument.")
    
    if args.inference_mode == "gradient_ascent":
        if args.num_steps is None:
            parser.error("The 'gradient_ascent' inference mode requires the --num-steps argument.")
        if args.lr is None:
            parser.error("The 'gradient_ascent' inference mode requires the --lr argument.")
    
    # Build inference mode kwargs
    inference_mode_kwargs = {
        "num_samples": args.num_samples,
        "scale": args.scale,
        "num_steps": args.num_steps,
        "lr": args.lr,
    }
    
    for arg in [
        "scan_batch_size",
        "include_mean_latent", 
        "include_all_latents",
        "lr_schedule",
        "lr_schedule_exponent",
        "optimizer",
        "optimizer_kwargs",
        "scan_gradients_latents",
        "accumulate_gradients_decoder_pairs",
        "random_perturbation",
    ]:
        if getattr(args, arg) is not None:
            inference_mode_kwargs[arg] = getattr(args, arg)
    
    main(
        artifact_path=args.wandb_artifact_path,
        json_challenges_file=args.json_challenges_file,
        json_solutions_file=args.json_solutions_file,
        only_n_tasks=args.only_n_tasks,
        dataset_folder=args.dataset_folder,
        dataset_length=args.dataset_length,
        dataset_batch_size=args.dataset_batch_size,
        dataset_use_hf=args.dataset_use_hf,
        dataset_seed=args.dataset_seed,
        inference_mode=args.inference_mode,
        inference_mode_kwargs=inference_mode_kwargs,
        random_search_seed=args.random_search_seed,
        mixed_precision=args.mixed_precision,
    )