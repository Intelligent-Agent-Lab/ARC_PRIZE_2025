import os
from typing import Optional, Tuple, List

import torch
import numpy as np

from huggingface_hub import hf_hub_download


DATASETS_BASE_PATH = "src/datasets"


def load_datasets(dataset_dirs: List[str], use_hf: bool) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load datasets from the given directories.

    Args:
        dataset_dirs: List of directories containing the datasets.
        use_hf: Whether to use the HF hub to download the datasets.

    Returns:
        List of tuples containing the grids and shapes of the datasets.
    """

    datasets = []

    if use_hf:
        for dataset_dir in dataset_dirs:
            grids = np.load(
                hf_hub_download(
                    repo_id="arcenv/arc_datasets",
                    filename=os.path.join(dataset_dir, "grids.npy"),
                    repo_type="dataset",
                )
            ).astype("uint8")
            shapes = np.load(
                hf_hub_download(
                    repo_id="arcenv/arc_datasets",
                    filename=os.path.join(dataset_dir, "shapes.npy"),
                    repo_type="dataset",
                )
            ).astype("uint8")
            try:
                program_ids = np.load(
                    hf_hub_download(
                        repo_id="arcenv/arc_datasets",
                        filename=os.path.join(dataset_dir, "program_ids.npy"),
                        repo_type="dataset",
                    )
                ).astype("uint32")
            except:
                program_ids = np.zeros(grids.shape[0], dtype=np.uint32)

            # Convert to PyTorch tensors
            grids = torch.from_numpy(grids)
            shapes = torch.from_numpy(shapes)
            program_ids = torch.from_numpy(program_ids)
            
            datasets.append((grids, shapes, program_ids))

    else:
        for dataset_dir in dataset_dirs:
            grids = np.load(os.path.join(DATASETS_BASE_PATH, dataset_dir, "grids.npy")).astype("uint8")
            shapes = np.load(os.path.join(DATASETS_BASE_PATH, dataset_dir, "shapes.npy")).astype("uint8")

            try:
                program_ids = np.load(
                    os.path.join(DATASETS_BASE_PATH, dataset_dir, "program_ids.npy")
                ).astype("uint32")
            except:
                program_ids = np.zeros(grids.shape[0], dtype=np.uint32)

            # Convert to PyTorch tensors
            grids = torch.from_numpy(grids)
            shapes = torch.from_numpy(shapes)
            program_ids = torch.from_numpy(program_ids)
            
            datasets.append((grids, shapes, program_ids))
    
    return datasets


def shuffle_dataset_into_batches(
    dataset_grids: torch.Tensor, 
    dataset_shapes: torch.Tensor, 
    batch_size: int, 
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shuffle dataset and organize into batches.
    
    Args:
        dataset_grids: Input grids tensor
        dataset_shapes: Input shapes tensor  
        batch_size: Size of each batch
        generator: Random number generator for reproducibility
        
    Returns:
        Tuple of batched grids and shapes
    """
    if dataset_grids.shape[0] != dataset_shapes.shape[0]:
        raise ValueError("Dataset grids and shapes must have the same length.")

    # Shuffle the dataset
    shuffled_indices = torch.randperm(len(dataset_grids), generator=generator)
    shuffled_grids = dataset_grids[shuffled_indices]
    shuffled_shapes = dataset_shapes[shuffled_indices]

    # Determine the number of batches
    num_batches = len(dataset_grids) // batch_size
    if num_batches < 1:
        raise ValueError(f"Got dataset size: {len(dataset_grids)} < batch size: {batch_size}.")

    # Reshape the dataset into batches and crop the last batch if necessary
    batched_grids = shuffled_grids[:num_batches * batch_size].view(
        num_batches, batch_size, *dataset_grids.shape[1:]
    )
    batched_shapes = shuffled_shapes[:num_batches * batch_size].view(
        num_batches, batch_size, *dataset_shapes.shape[1:]
    )

    return batched_grids, batched_shapes


def _apply_rotation(grid: torch.Tensor, grid_shape: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply k 90-degree counter-clockwise rotations to a grid.
    
    Args:
        grid: 2D grid tensor
        grid_shape: Shape of the grid [rows, cols]
        k: Number of 90-degree rotations to apply
        
    Returns:
        Rotated grid and updated shape
    """
    assert grid.ndim == 2 and grid_shape.ndim == 1

    for _ in range(k % 4):  # Only need to apply up to 3 rotations
        # Rotate 90 degrees counter-clockwise (equivalent to rot90 with k=-1)
        grid = torch.rot90(grid, k=1)  # PyTorch rot90 with k=1 rotates counter-clockwise
        
        # Roll the columns to the left until the first non-padded column is at the first position
        num_rows = grid_shape[0].int()
        for _ in range(grid.shape[0] - num_rows):
            grid = torch.roll(grid, shifts=-1, dims=-1)
        
        # Swap the rows and cols in shape
        grid_shape = torch.flip(grid_shape, dims=[0])

    return grid, grid_shape


def _apply_color_permutation(grid: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Apply random color permutation to grid, exempting black (0).
    
    Args:
        grid: Input grid tensor
        generator: Random number generator for reproducibility
        
    Returns:
        Grid with permuted colors
    """
    device = grid.device
    dtype = grid.dtype
    
    # Exempt black (0)
    non_exempt = torch.tensor([i for i in range(10) if i != 0], dtype=dtype, device=device)
    
    # Create random permutation
    permutation = torch.randperm(len(non_exempt), generator=generator, device=device)
    permuted_non_exempt = non_exempt[permutation]
    
    # Create color mapping
    color_map = torch.arange(10, dtype=dtype, device=device)
    color_map[non_exempt] = permuted_non_exempt
    
    return color_map[grid]


def data_augmentation_fn(
    grids: torch.Tensor, 
    shapes: torch.Tensor, 
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply data augmentation to the grids and shapes.

    Args:
        grids: The input grids. Shape (*B, N, R, C, 2).
        shapes: The shapes of the grids. Shape (*B, N, 2, 2).
        generator: Random number generator for reproducibility.

    Returns:
        The augmented grids and shapes.
    """
    device = grids.device
    
    # Generate random seeds for rotation and color permutation
    if generator is not None:
        rotation_seed = torch.randint(0, 2**32, (1,), generator=generator).item()
        color_seed = torch.randint(0, 2**32, (1,), generator=generator).item()
        rotation_gen = torch.Generator(device=device).manual_seed(rotation_seed)
        color_gen = torch.Generator(device=device).manual_seed(color_seed)
    else:
        rotation_gen = None
        color_gen = None

    # Generate rotation indices for each batch element
    batch_shape = grids.shape[:-4]
    rotation_indices = torch.randint(0, 4, batch_shape, generator=rotation_gen, device=device)

    # Apply rotations
    augmented_grids = grids.clone()
    augmented_shapes = shapes.clone()
    
    # We need to iterate through all batch dimensions and apply rotations
    def apply_rotation_recursive(grids_batch, shapes_batch, rotation_idx):
        if grids_batch.ndim == 4:  # Base case: (N, R, C, 2)
            # Apply same rotation to all pairs and both input/output channels
            rotated_grids = grids_batch.clone()
            rotated_shapes = shapes_batch.clone()
            
            for pair_idx in range(grids_batch.shape[0]):  # N pairs
                for channel_idx in range(grids_batch.shape[3]):  # 2 channels
                    grid = grids_batch[pair_idx, :, :, channel_idx]
                    shape = shapes_batch[pair_idx, :, channel_idx]
                    rotated_grid, rotated_shape = _apply_rotation(grid, shape, rotation_idx.item())
                    rotated_grids[pair_idx, :, :, channel_idx] = rotated_grid
                    rotated_shapes[pair_idx, :, channel_idx] = rotated_shape
            
            return rotated_grids, rotated_shapes
        else:
            # Recursive case: handle additional batch dimensions
            result_grids = []
            result_shapes = []
            for i in range(grids_batch.shape[0]):
                r_grids, r_shapes = apply_rotation_recursive(
                    grids_batch[i], shapes_batch[i], rotation_idx[i]
                )
                result_grids.append(r_grids)
                result_shapes.append(r_shapes)
            return torch.stack(result_grids), torch.stack(result_shapes)
    
    if len(batch_shape) > 0:
        augmented_grids, augmented_shapes = apply_rotation_recursive(
            augmented_grids, augmented_shapes, rotation_indices
        )

    # Apply color permutation
    def apply_color_permutation_recursive(grids_batch, batch_dims_left):
        if batch_dims_left == 0:
            # Base case: apply color permutation
            return _apply_color_permutation(grids_batch, color_gen)
        else:
            # Recursive case: handle batch dimensions
            result = []
            for i in range(grids_batch.shape[0]):
                # Generate new generator for each batch element for reproducibility
                if color_gen is not None:
                    element_seed = torch.randint(0, 2**32, (1,), generator=color_gen).item()
                    element_gen = torch.Generator(device=device).manual_seed(element_seed)
                else:
                    element_gen = None
                
                result.append(apply_color_permutation_recursive(
                    grids_batch[i], batch_dims_left - 1
                ))
            return torch.stack(result)
    
    if len(batch_shape) > 0:
        augmented_grids = apply_color_permutation_recursive(augmented_grids, len(batch_shape))
    else:
        augmented_grids = _apply_color_permutation(augmented_grids, color_gen)

    return augmented_grids, augmented_shapes


def make_leave_one_out(array: torch.Tensor, axis: int) -> torch.Tensor:
    """Create leave-one-out version of array.
    
    Args:
        array: Input tensor of shape (*B, N, *H).
        axis: The axis where N appears.

    Returns:
        Tensor of shape (*B, N, N-1, *H).
    """
    axis = axis % array.ndim
    output = []
    
    for i in range(array.shape[axis]):
        # Create slices for before and after the i-th element
        indices_before = torch.arange(0, i, device=array.device)
        indices_after = torch.arange(i + 1, array.shape[axis], device=array.device)
        
        # Combine indices
        if len(indices_before) > 0 and len(indices_after) > 0:
            combined_indices = torch.cat([indices_before, indices_after])
        elif len(indices_before) > 0:
            combined_indices = indices_before
        elif len(indices_after) > 0:
            combined_indices = indices_after
        else:
            # This case shouldn't happen if N > 1, but handle it gracefully
            combined_indices = torch.empty(0, dtype=torch.long, device=array.device)
        
        # Use advanced indexing to select elements
        array_without_i = torch.index_select(array, axis, combined_indices)
        output.append(array_without_i)
    
    # Stack along the specified axis
    output = torch.stack(output, dim=axis)
    return output


# Test functions
def test_data_utils():
    """Test the data utility functions."""
    print("Testing data utility functions...")
    
    # Test shuffle_dataset_into_batches
    grids = torch.randn(10, 3, 30, 30, 2)
    shapes = torch.randint(1, 31, (10, 3, 2, 2))
    
    batched_grids, batched_shapes = shuffle_dataset_into_batches(grids, shapes, batch_size=4)
    print(f"Original grids shape: {grids.shape}")
    print(f"Batched grids shape: {batched_grids.shape}")
    assert batched_grids.shape == (2, 4, 3, 30, 30, 2)
    
    # Test _apply_rotation
    grid = torch.randint(0, 10, (5, 5))
    shape = torch.tensor([3, 4])
    rotated_grid, rotated_shape = _apply_rotation(grid, shape, k=1)
    print(f"Original shape: {shape}, Rotated shape: {rotated_shape}")
    
    # Test _apply_color_permutation
    grid = torch.randint(0, 10, (5, 5))
    permuted_grid = _apply_color_permutation(grid)
    print("Color permutation applied successfully")
    
    # Test data_augmentation_fn
    grids = torch.randint(0, 10, (2, 3, 10, 10, 2))
    shapes = torch.randint(5, 10, (2, 3, 2, 2))
    
    aug_grids, aug_shapes = data_augmentation_fn(grids, shapes)
    print(f"Data augmentation: {grids.shape} -> {aug_grids.shape}")
    assert aug_grids.shape == grids.shape
    assert aug_shapes.shape == shapes.shape
    
    # Test make_leave_one_out
    array = torch.randn(4, 5, 10)
    loo_array = make_leave_one_out(array, axis=1)
    print(f"Leave-one-out: {array.shape} -> {loo_array.shape}")
    assert loo_array.shape == (4, 5, 4, 10)
    
    print("All tests passed!")


if __name__ == "__main__":
    test_data_utils()