from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np

# Assume these imports are replaced with your PyTorch versions
# from src.models.lpn import LPN
# from src.datasets.task_gen.re_arc_generators import ARC_TASK_NAMES

# Mock imports for demonstration
class LPN(nn.Module):
    def __init__(self):
        super().__init__()
        # Mock implementation
        pass

ARC_TASK_NAMES = []  # Replace with actual task names


class Evaluator:
    def __init__(
        self, 
        model: LPN, 
        inference_mode: str, 
        inference_mode_kwargs: dict, 
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.inference_mode = inference_mode
        self.inference_mode_kwargs = inference_mode_kwargs
        self.max_rows = self.model.encoder.config.max_rows
        self.max_cols = self.model.encoder.config.max_cols
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug_msg = False
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def json_submission(
        self,
        challenges: Dict[str, List],
        only_n_tasks: Optional[int] = None,
        overfit_task: Optional[str] = None,
        progress_bar: bool = False,
        train: bool = False,
    ) -> Dict[str, List]:
        """
        Generate solutions for the given challenges.
        
        Args:
            challenges: Dictionary of tasks with their examples
            only_n_tasks: Limit the number of tasks to evaluate
            overfit_task: Evaluate only a specific task
            progress_bar: Show progress bar
            train: Whether this is training data
            
        Returns:
            Dictionary with task results
        """
        assert only_n_tasks is None or overfit_task is None, "Cannot use both only_n_tasks and overfit_task."
        
        if overfit_task is not None:
            assert overfit_task in challenges, f"Task {overfit_task} not found in the challenges."
            challenges = {overfit_task: challenges[overfit_task]}
            only_n_tasks = None
            
        if only_n_tasks is not None:
            num_tasks = min(only_n_tasks, len(challenges))
        else:
            num_tasks = len(challenges)
            
        if num_tasks < len(challenges):
            task_names = ARC_TASK_NAMES if train else list(challenges.keys())
            challenges = {task_name: challenges[task_name] for task_name in task_names[:num_tasks]}

        results = {}
        
        with torch.no_grad():  # Disable gradients for inference
            for task_id, task in tqdm(
                challenges.items(), total=num_tasks, desc="Generating solutions", disable=not progress_bar
            ):
                pair_list, shape_list = [], []
                
                # Process training examples
                for example in task["train"]:
                    input_array = np.array(example["input"])
                    input_shape = input_array.shape
                    input_tensor, input_shape = self.pad_and_crop_json(
                        torch.from_numpy(input_array), input_shape
                    )
                    
                    output_array = np.array(example["output"])
                    output_shape = output_array.shape
                    output_tensor, output_shape = self.pad_and_crop_json(
                        torch.from_numpy(output_array), output_shape
                    )
                    
                    pair_list.append(torch.stack([input_tensor, output_tensor], dim=-1))
                    shape_list.append(torch.tensor([
                        [input_shape[0], output_shape[0]],
                        [input_shape[1], output_shape[1]]
                    ], dtype=torch.long))
                
                pairs = torch.stack(pair_list).to(self.device)
                grid_shapes = torch.stack(shape_list).to(self.device)

                task_outputs = []
                
                # Process test examples
                for example in task["test"]:
                    input_array = np.array(example["input"])
                    input_tensor, input_grid_shape = self.pad_and_crop_json(
                        torch.from_numpy(input_array), input_array.shape
                    )
                    input_grid_shape = torch.tensor(input_grid_shape, dtype=torch.long)
                    
                    # Move tensors to device and add batch dimension
                    input_tensor = input_tensor.unsqueeze(0).to(self.device)
                    input_grid_shape = input_grid_shape.unsqueeze(0).to(self.device)
                    pairs_batch = pairs.unsqueeze(0)
                    grid_shapes_batch = grid_shapes.unsqueeze(0)
                    
                    # Generate outputs
                    outputs = self.model.generate_output(
                        pairs_batch,
                        grid_shapes_batch,
                        input_tensor,
                        input_grid_shape,
                        training=False,
                        mode=self.inference_mode,
                        return_two_best=True,
                        **self.inference_mode_kwargs
                    )
                    
                    if len(outputs) == 5:  # return_two_best=True
                        first_output_grid, first_output_grid_shape, second_output_grid, second_output_grid_shape, _ = outputs
                    else:  # return_two_best=False
                        first_output_grid, first_output_grid_shape, _ = outputs
                        second_output_grid, second_output_grid_shape = first_output_grid, first_output_grid_shape
                    
                    # Remove batch dimension and move to CPU
                    first_output_grid = first_output_grid.squeeze(0).cpu()
                    first_output_grid_shape = first_output_grid_shape.squeeze(0).cpu()
                    second_output_grid = second_output_grid.squeeze(0).cpu()
                    second_output_grid_shape = second_output_grid_shape.squeeze(0).cpu()

                    # Crop the output to the predicted shape
                    first_num_rows, first_num_cols = first_output_grid_shape.tolist()
                    second_num_rows, second_num_cols = second_output_grid_shape.tolist()
                    
                    attempts = {
                        "attempt_1": first_output_grid[:first_num_rows, :first_num_cols].tolist(),
                        "attempt_2": second_output_grid[:second_num_rows, :second_num_cols].tolist(),
                    }
                    task_outputs.append(attempts)
                    
                results[task_id] = task_outputs
                
        return results

    def evaluate_generations(
        self, generations: Dict[str, List], solutions: Dict[str, List]
    ) -> Dict[str, float]:
        """
        Evaluate the quality of generated solutions.
        
        Args:
            generations: Generated solutions
            solutions: Ground truth solutions
            
        Returns:
            Dictionary with evaluation metrics
        """
        top_1_num_correct_tasks, top_2_num_correct_tasks = 0.0, 0.0
        top_1_num_correct_shapes, top_2_num_correct_shapes = 0.0, 0.0
        top_1_pixel_correctness, top_2_pixel_correctness = 0.0, 0.0
        
        for task_id, generation_outputs in generations.items():
            num_test_grids = len(generation_outputs)
            
            for generation, solution in zip(generation_outputs, solutions[task_id]):
                attempt_1 = np.array(generation["attempt_1"])
                attempt_2 = np.array(generation["attempt_2"])
                solution = np.array(solution)
                
                maybe_top_2_num_correct_shapes = 0
                maybe_top_2_num_correct_tasks = 0
                maybe_top_2_pixel_correctness = 0
                
                # Evaluate first attempt
                if attempt_1.shape == solution.shape:
                    top_1_num_correct_shapes += 1 / num_test_grids
                    task_correct = np.array_equal(attempt_1, solution)
                    top_1_num_correct_tasks += task_correct / num_test_grids
                    pixel_correct = np.mean(attempt_1 == solution).item()
                    top_1_pixel_correctness += pixel_correct / num_test_grids
                    
                    maybe_top_2_num_correct_shapes = 1 / num_test_grids
                    maybe_top_2_num_correct_tasks = task_correct / num_test_grids
                    maybe_top_2_pixel_correctness = pixel_correct / num_test_grids
                
                # Evaluate second attempt
                if attempt_2.shape == solution.shape:
                    maybe_top_2_num_correct_shapes = 1 / num_test_grids
                    task_correct_2 = np.array_equal(attempt_2, solution) / num_test_grids
                    pixel_correct_2 = np.mean(attempt_2 == solution).item() / num_test_grids
                    
                    maybe_top_2_num_correct_tasks = max(task_correct_2, maybe_top_2_num_correct_tasks)
                    maybe_top_2_pixel_correctness = max(pixel_correct_2, maybe_top_2_pixel_correctness)
                
                top_2_num_correct_shapes += maybe_top_2_num_correct_shapes
                top_2_num_correct_tasks += maybe_top_2_num_correct_tasks
                top_2_pixel_correctness += maybe_top_2_pixel_correctness
        
        num_tasks = len(generations)
        metrics = {
            "top_1_shape_accuracy": top_1_num_correct_shapes / num_tasks,
            "top_1_accuracy": top_1_num_correct_tasks / num_tasks,
            "top_1_pixel_correctness": top_1_pixel_correctness / num_tasks,
            "top_2_shape_accuracy": top_2_num_correct_shapes / num_tasks,
            "top_2_accuracy": top_2_num_correct_tasks / num_tasks,
            "top_2_pixel_correctness": top_2_pixel_correctness / num_tasks,
        }
        return metrics

    def pad_and_crop_json(
        self, x: torch.Tensor, x_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Pad and crop tensor to fit model requirements.
        
        Args:
            x: Input tensor
            x_shape: Original shape
            
        Returns:
            Processed tensor and adjusted shape
        """
        if x.shape[0] > self.max_rows or x.shape[1] > self.max_cols:
            if not self.debug_msg:
                print(
                    f"WARNING: cropping json grids to {self.max_rows, self.max_cols}. "
                    "The outputs cannot be trusted."
                )
                self.debug_msg = True
            x = x[:self.max_rows, :self.max_cols]
            # Clamp the shape to the max values
            x_shape = (min(x_shape[0], self.max_rows), min(x_shape[1], self.max_cols))
        
        # Pad tensor to max dimensions
        pad_rows = self.max_rows - x.shape[0]
        pad_cols = self.max_cols - x.shape[1]
        x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows), value=0)
        
        return x, x_shape

    def set_device(self, device: torch.device):
        """Set the device for the evaluator and move model."""
        self.device = device
        self.model.to(device)

    def cuda(self):
        """Move evaluator to CUDA device."""
        if torch.cuda.is_available():
            self.set_device(torch.device('cuda'))
        else:
            print("CUDA not available, staying on CPU")

    def cpu(self):
        """Move evaluator to CPU device."""
        self.set_device(torch.device('cpu'))


# Example usage and test
if __name__ == "__main__":
    # Mock model for testing
    class MockEncoder:
        def __init__(self):
            self.config = type('Config', (), {'max_rows': 30, 'max_cols': 30})()
    
    class MockLPN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MockEncoder()
        
        def generate_output(self, pairs, grid_shapes, input_tensor, input_grid_shape, 
                          training=False, mode="mean", return_two_best=False, **kwargs):
            batch_size = input_tensor.shape[0]
            if return_two_best:
                return (
                    torch.randint(0, 10, (batch_size, 30, 30)),  # first_output_grid
                    torch.randint(1, 31, (batch_size, 2)),       # first_output_grid_shape
                    torch.randint(0, 10, (batch_size, 30, 30)),  # second_output_grid
                    torch.randint(1, 31, (batch_size, 2)),       # second_output_grid_shape
                    {}  # info
                )
            else:
                return (
                    torch.randint(0, 10, (batch_size, 30, 30)),  # output_grid
                    torch.randint(1, 31, (batch_size, 2)),       # output_grid_shape
                    {}  # info
                )
    
    # Test the evaluator
    model = MockLPN()
    evaluator = Evaluator(
        model=model,
        inference_mode="mean",
        inference_mode_kwargs={},
        device=torch.device('cpu')
    )
    
    # Test pad_and_crop_json
    test_tensor = torch.randint(0, 10, (10, 15))
    padded_tensor, new_shape = evaluator.pad_and_crop_json(test_tensor, (10, 15))
    print(f"Original shape: {test_tensor.shape}")
    print(f"Padded shape: {padded_tensor.shape}")
    print(f"New shape: {new_shape}")
    
    # Mock challenge data
    mock_challenges = {
        "task_1": {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[2, 3], [4, 5]]
                }
            ],
            "test": [
                {
                    "input": [[1, 2], [3, 4]]
                }
            ]
        }
    }
    
    # Test json_submission
    results = evaluator.json_submission(mock_challenges, progress_bar=True)
    print(f"Results: {list(results.keys())}")
    print(f"Task 1 outputs: {len(results['task_1'])}")
    
    # Test evaluation
    mock_solutions = {
        "task_1": [[[2, 3], [4, 5]]]
    }
    
    metrics = evaluator.evaluate_generations(results, mock_solutions)
    print(f"Evaluation metrics: {metrics}")