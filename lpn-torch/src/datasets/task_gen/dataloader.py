from typing import Any, Iterator, Literal, Optional, Tuple, Dict, List

import torch
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from tqdm.auto import trange
from functools import partial

class TaskGeneratorIterableDataset(IterableDataset):
    def __init__(self, task_generator):
        self.task_generator = task_generator

    def __iter__(self):
        return iter(self.task_generator)

# Import your PyTorch modules (replace with actual imports)
# from src.datasets.task_gen.task_generator import PatternTaskGenerator, ArcTrainTaskGenerator
# from src.data_utils import data_augmentation_fn


# Mock imports for demonstration
class PatternTaskGenerator:
    def __init__(self, num_pairs: int, seed: Optional[int] = None, **kwargs):
        self.num_pairs = num_pairs
        self.num_rows = kwargs.get("max_rows", 30)
        self.num_cols = kwargs.get("max_cols", 30)
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def __iter__(self):
        while True:
            yield self._generate_task()
    
    def _generate_task(self):
        task = []
        for _ in range(self.num_pairs):
            input_grid = np.random.randint(0, 10, (np.random.randint(5, 15), np.random.randint(5, 15)))
            output_grid = np.random.randint(0, 10, (np.random.randint(5, 15), np.random.randint(5, 15)))
            task.append({"input": input_grid, "output": output_grid})
        return task, {"num_attempts_generate_task": 1, "program_id": np.random.randint(0, 1000)}

class ArcTrainTaskGenerator:
    def __init__(self, num_pairs: int, seed: Optional[int] = None, **kwargs):
        self.num_pairs = num_pairs
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def __iter__(self):
        while True:
            yield self._generate_task()
    
    def _generate_task(self):
        task = []
        for _ in range(self.num_pairs):
            input_grid = np.random.randint(0, 10, (np.random.randint(5, 15), np.random.randint(5, 15)))
            output_grid = np.random.randint(0, 10, (np.random.randint(5, 15), np.random.randint(5, 15)))
            task.append({"input": input_grid, "output": output_grid})
        return task, {"num_attempts_generate_task": 1, "program_id": np.random.randint(0, 1000)}

def data_augmentation_fn(grids: torch.Tensor, shapes: torch.Tensor, generator: Optional[torch.Generator] = None):
    return grids, shapes


class TorchDataLoader:
    """DataLoader wrapper for PyTorch with JAX-like functionality."""

    def __init__(
        self,
        task_generator: PatternTaskGenerator | ArcTrainTaskGenerator,
        batch_size: int,
        log_every_n_steps: int,
        num_workers: int,
        worker_timeout: int = 0,
        device: Optional[torch.device] = None,
        online_data_augmentation: bool = True,
        seed: Optional[int] = None,
        return_info: bool = False,
        max_rows: int = 30,
        max_cols: int = 30,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        dataset = TaskGeneratorIterableDataset(task_generator)
        self.numpy_dataloader = DataLoader(
            dataset,
            batch_size=batch_size * log_every_n_steps,
            num_workers=num_workers,
            timeout=worker_timeout,
            collate_fn=partial(
                collate_fn,
                batch_size=batch_size,
                log_every_n_steps=log_every_n_steps,
                device=self.device,
                return_info=return_info,
                max_rows=max_rows,
                max_cols=max_cols,
            ),
        )

        
        self.online_data_augmentation = online_data_augmentation
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)
        self.return_info = return_info

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict]]:
        for batch in self.numpy_dataloader:
            if self.online_data_augmentation:
                if self.return_info:
                    if len(batch) == 3:
                        grids, shapes, info = batch
                    else:
                        grids, shapes = batch
                        info = {}
                    grids, shapes = data_augmentation_fn(grids, shapes, self.generator)
                    yield (grids, shapes, info)
                else:
                    grids, shapes = batch
                    grids, shapes = data_augmentation_fn(grids, shapes, self.generator)
                    yield (grids, shapes)
            else:
                if self.return_info:
                    if len(batch) == 3:
                        yield batch  # already (grids, shapes, info)
                    else:
                        grids, shapes = batch
                        info = {}
                        yield (grids, shapes, info)
                else:
                    yield batch  # already (grids, shapes)


def collate_fn(
    batch: List[Tuple[List[Dict[str, tuple]], Dict[str, Any]]],
    batch_size: int,
    log_every_n_steps: int,
    device: Optional[torch.device] = None,
    return_info: bool = False,
    max_rows: int = 30,
    max_cols: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Collate function to process batches of tasks."""
    
    tasks, infos = zip(*batch)
    grids, shapes = [], []
    
    for task in tasks:
        task_input_grids, task_input_shapes, task_output_grids, task_output_shapes = [], [], [], []
        for pair in task:
            input_grid, output_grid = pair["input"], pair["output"]
            input_shape, output_shape = input_grid.shape, output_grid.shape
            
            # Pad grids to max dimensions
            input_grid = np.pad(
                input_grid, 
                ((0, max_rows - input_shape[0]), (0, max_cols - input_shape[1]))
            )
            output_grid = np.pad(
                output_grid, 
                ((0, max_rows - output_shape[0]), (0, max_cols - output_shape[1]))
            )
            
            task_input_grids.append(input_grid)
            task_input_shapes.append(input_shape)
            task_output_grids.append(output_grid)
            task_output_shapes.append(output_shape)
        
        task_input_grids = np.stack(task_input_grids)
        task_input_shapes = np.stack(task_input_shapes)
        task_output_grids = np.stack(task_output_grids)
        task_output_shapes = np.stack(task_output_shapes)
        
        # Stack input and output grids along last dimension
        grids.append(np.stack([task_input_grids, task_output_grids], axis=-1))
        shapes.append(np.stack([task_input_shapes, task_output_shapes], axis=-1))
    
    grids, shapes = np.stack(grids), np.stack(shapes)
    
    # Reshape for training loop structure
    grids = grids.reshape(log_every_n_steps, batch_size, *grids.shape[1:])
    shapes = shapes.reshape(log_every_n_steps, batch_size, *shapes.shape[1:])

    # numpy -> torch 변환
    grids = torch.from_numpy(grids).to(dtype=torch.uint8, device=device)
    shapes = torch.from_numpy(shapes).to(dtype=torch.uint8, device=device)

    if return_info:
        # Process info dictionary
        num_attempts = np.array([info["num_attempts_generate_task"] for info in infos])
        info = {"num_attempts_generate_task": num_attempts}
        
        if "program_id" in infos[0]:
            program_ids = np.array([info["program_id"] for info in infos])
            info["program_ids"] = program_ids
        
        # Reshape info to match grids/shapes structure and convert to torch
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                value = value.reshape(log_every_n_steps, batch_size, *value.shape[1:])
                info[key] = torch.from_numpy(value).to(device=device)
        
        return grids, shapes, info
    
    return grids, shapes


def make_task_gen_dataloader(
    batch_size: int,
    log_every_n_steps: int,
    num_workers: int,
    task_generator_class: Literal["PATTERN", "ARC"],
    num_pairs: int,
    worker_timeout: int = 0,
    device: Optional[torch.device] = None,
    online_data_augmentation: bool = True,
    return_info: bool = False,
    seed: Optional[int] = None,
    **task_generator_kwargs,
) -> TorchDataLoader:
    """Create a task generator dataloader.
    
    Args:
        batch_size: Size of each batch
        log_every_n_steps: Number of steps between logging
        num_workers: Number of worker processes for data loading
        task_generator_class: Type of task generator ("PATTERN" or "ARC")
        num_pairs: Number of input-output pairs per task
        worker_timeout: Timeout for worker processes
        device: PyTorch device to use
        online_data_augmentation: Whether to apply data augmentation
        return_info: Whether to return additional info
        seed: Random seed
        **task_generator_kwargs: Additional arguments for task generator
        
    Returns:
        TorchDataLoader instance
    """
    max_rows = task_generator_kwargs.get("max_rows", 30)
    max_cols = task_generator_kwargs.get("max_cols", 30)
    
    if task_generator_class == "PATTERN":
        task_generator = PatternTaskGenerator(num_pairs=num_pairs, seed=seed, **task_generator_kwargs)
        max_rows, max_cols = task_generator.num_rows, task_generator.num_cols
    elif task_generator_class == "ARC":
        task_generator = ArcTrainTaskGenerator(num_pairs=num_pairs, seed=seed, **task_generator_kwargs)
    else:
        raise ValueError(f"Invalid task_generator_class: {task_generator_class}")
    
    torch_dataloader = TorchDataLoader(
        task_generator,
        batch_size,
        log_every_n_steps,
        num_workers,
        worker_timeout,
        device,
        online_data_augmentation,
        seed,
        return_info,
        max_rows,
        max_cols,
    )
    return torch_dataloader


def make_dataset(
    length: int,
    num_pairs: int,
    num_workers: int,
    task_generator_class: Literal["PATTERN", "ARC"],
    online_data_augmentation: bool = True,
    seed: int = 0,
    device: Optional[torch.device] = None,
    **task_generator_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a dataset using task generators.
    
    Args:
        length: Number of tasks to generate
        num_pairs: Number of input-output pairs per task
        num_workers: Number of worker processes
        task_generator_class: Type of task generator
        online_data_augmentation: Whether to apply data augmentation
        seed: Random seed
        device: PyTorch device
        **task_generator_kwargs: Additional task generator arguments
        
    Returns:
        Tuple of (grids, shapes, program_ids)
    """
    dataloader = make_task_gen_dataloader(
        batch_size=1,
        log_every_n_steps=1,
        num_workers=num_workers,
        task_generator_class=task_generator_class,
        num_pairs=num_pairs,
        device=device,
        online_data_augmentation=online_data_augmentation,
        return_info=True,
        seed=seed,
        **task_generator_kwargs,
    )
    
    dataset_grids, dataset_shapes, program_ids = [], [], []
    
    for (grids, shapes, info), _ in zip(dataloader, trange(length, desc="Generating dataset")):
        dataset_grids.append(grids[0, 0])  # Extract single sample from batch
        dataset_shapes.append(shapes[0, 0])
        program_ids.append(info["program_ids"][0, 0] if "program_ids" in info else 0)
    
    # Stack all samples
    dataset_grids = torch.stack(dataset_grids)
    dataset_shapes = torch.stack(dataset_shapes)
    program_ids = torch.stack([torch.tensor(pid, dtype=torch.uint32) for pid in program_ids])
    
    del dataloader
    return dataset_grids, dataset_shapes, program_ids


class DataLoaderBenchmark:
    """Utility class for benchmarking dataloader performance."""
    
    def __init__(self, dataloader: TorchDataLoader):
        self.dataloader = dataloader
    
    def benchmark_throughput(self, num_batches: int = 100) -> Dict[str, float]:
        """Benchmark dataloader throughput.
        
        Args:
            num_batches: Number of batches to process for benchmarking
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        times = []
        total_samples = 0
        
        start_time = time.time()
        for i, batch in enumerate(self.dataloader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            if isinstance(batch, tuple) and len(batch) >= 2:
                grids = batch[0]
                batch_size = grids.shape[0] * grids.shape[1] if grids.ndim > 2 else grids.shape[0]
                total_samples += batch_size
            
            batch_time = time.time() - batch_start
            times.append(batch_time)
        
        total_time = time.time() - start_time
        
        return {
            "total_time": total_time,
            "avg_batch_time": np.mean(times),
            "std_batch_time": np.std(times),
            "total_samples": total_samples,
            "samples_per_second": total_samples / total_time if total_time > 0 else 0,
        }


# Test and utility functions
def test_dataloader():
    """Test the dataloader functionality."""
    print("Testing PyTorch DataLoader...")
    
    # Test dataloader creation
    dataloader = make_task_gen_dataloader(
        batch_size=4,
        log_every_n_steps=2,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        task_generator_class="PATTERN",
        num_pairs=3,
        seed=42,
        return_info=True,
    )
    
    print(f"Created dataloader: {type(dataloader)}")
    
    # Test a few batches
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Test first 3 batches
            break
        
        if len(batch) == 3:  # With info
            grids, shapes, info = batch
            print(f"Batch {i}: grids={grids.shape}, shapes={shapes.shape}, info_keys={list(info.keys())}")
        else:  # Without info
            grids, shapes = batch
            print(f"Batch {i}: grids={grids.shape}, shapes={shapes.shape}")
    
    print("DataLoader test completed successfully!")


def test_dataset_generation():
    """Test dataset generation functionality."""
    print("Testing dataset generation...")
    
    grids, shapes, program_ids = make_dataset(
        length=10,
        num_pairs=3,
        num_workers=0,
        task_generator_class="ARC",
        seed=42,
    )
    
    print("Generated dataset:")
    print(f"  Grids shape: {grids.shape}")
    print(f"  Shapes shape: {shapes.shape}")
    print(f"  Program IDs shape: {program_ids.shape}")
    print(f"  Device: {grids.device}")
    
    print("Dataset generation test completed successfully!")


if __name__ == "__main__":
    import time
    
    # Mock EMA class for testing
    class EMA:
        def __init__(self, start: float, smoothing: float = 0.05, return_inverse: bool = False):
            self.start = start
            self.smoothing = smoothing
            self.return_inverse = return_inverse
            self.value = None
        
        def __call__(self, current_time: float) -> float:
            if self.value is None:
                self.value = 1.0 / (current_time - self.start + 1e-8)
            else:
                current_rate = 1.0 / (current_time - self.start + 1e-8)
                self.value = self.smoothing * current_rate + (1 - self.smoothing) * self.value
            return self.value
    
    # Run tests
    test_dataloader()
    print("\n" + "="*50 + "\n")
    test_dataset_generation()
    print("\n" + "="*50 + "\n")
    
    # Performance benchmark
    print("Running performance benchmark...")
    
    dataloader = make_task_gen_dataloader(
        batch_size=100,
        log_every_n_steps=1,
        num_workers=0,
        task_generator_class="ARC",
        num_pairs=4,
        seed=0,
    )
    
    ema = EMA(start=time.time(), smoothing=0.05, return_inverse=True)
    
    for (grids, shapes), i in zip(dataloader, range(10)):  # Test first 10 batches
        print(f"\nBatch {i + 1}")
        print(f"Grids shape: {grids.shape}")
        print(f"Device: {grids.device}")
        throughput = ema(time.time()) * grids.shape[0] * grids.shape[1]
        print(f"Throughput: {throughput:.2f} samples/s")
    
    # Benchmark with utility class
    print("\n" + "="*50 + "\n")
    print("Running detailed benchmark...")
    
    benchmark = DataLoaderBenchmark(dataloader)
    results = benchmark.benchmark_throughput(num_batches=20)
    
    print("Benchmark Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")