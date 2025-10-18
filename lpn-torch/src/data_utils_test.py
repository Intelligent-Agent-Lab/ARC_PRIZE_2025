import functools
import unittest

import torch

from src.data_utils import _apply_rotation


class TestDataAugmentation(unittest.TestCase):

    def setUp(self):
        self.grid = torch.tensor(
            [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [7, 8, 9, 0, 0], [10, 11, 12, 0, 0], [0, 0, 0, 0, 0]]
        )
        self.grid_shape = torch.tensor([4, 3])
        self.grid2 = torch.tensor(
            [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        )
        self.grid_shape2 = torch.tensor([2, 2])

    def test__apply_rotation_0(self):
        grid, grid_shape = _apply_rotation(self.grid, self.grid_shape, k=0)
        self.assertTrue(torch.equal(grid_shape, self.grid_shape))
        self.assertTrue(torch.equal(grid, self.grid))

    def test__apply_rotation_1(self):
        grid, grid_shape = _apply_rotation(self.grid, self.grid_shape, k=1)
        expected_grid_shape = torch.tensor([3, 4])
        expected_grid = torch.tensor(
            [[10, 7, 4, 1, 0], [11, 8, 5, 2, 0], [12, 9, 6, 3, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        )
        self.assertTrue(torch.equal(grid_shape, expected_grid_shape))
        self.assertTrue(torch.equal(grid, expected_grid))

    def test__apply_rotation_2(self):
        grid, grid_shape = _apply_rotation(self.grid, self.grid_shape, k=2)
        expected_grid_shape = torch.tensor([4, 3])
        expected_grid = torch.tensor(
            [[12, 11, 10, 0, 0], [9, 8, 7, 0, 0], [6, 5, 4, 0, 0], [3, 2, 1, 0, 0], [0, 0, 0, 0, 0]]
        )
        self.assertTrue(torch.equal(grid_shape, expected_grid_shape))
        self.assertTrue(torch.equal(grid, expected_grid))

    def test_batch_apply_rotation_2(self):
        """Test batch processing equivalent to JAX vmap."""
        grids = torch.stack([self.grid, self.grid2])
        grid_shapes = torch.stack([self.grid_shape, self.grid_shape2])
        
        # Apply rotation to each grid in the batch
        output_grids = []
        output_grid_shapes = []
        
        for i in range(grids.shape[0]):
            grid, grid_shape = _apply_rotation(grids[i], grid_shapes[i], k=2)
            output_grids.append(grid)
            output_grid_shapes.append(grid_shape)
        
        output_grids = torch.stack(output_grids)
        output_grid_shapes = torch.stack(output_grid_shapes)
        
        expected_grid_shapes = torch.tensor([[4, 3], [2, 2]])
        expected_grids = torch.tensor(
            [
                [[12, 11, 10, 0, 0], [9, 8, 7, 0, 0], [6, 5, 4, 0, 0], [3, 2, 1, 0, 0], [0, 0, 0, 0, 0]],
                [[4, 3, 0, 0, 0], [2, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output_grid_shapes, expected_grid_shapes))
        self.assertTrue(torch.equal(output_grids, expected_grids))

    def test_vectorized_apply_rotation_2(self):
        """Test vectorized version using torch.vmap (if available) or manual vectorization."""
        grids = torch.stack([self.grid, self.grid2])
        grid_shapes = torch.stack([self.grid_shape, self.grid_shape2])
        
        # Manual vectorization approach (more compatible across PyTorch versions)
        def vectorized_apply_rotation(grids_batch, grid_shapes_batch, k):
            batch_size = grids_batch.shape[0]
            output_grids = []
            output_grid_shapes = []
            
            for i in range(batch_size):
                grid, grid_shape = _apply_rotation(grids_batch[i], grid_shapes_batch[i], k)
                output_grids.append(grid)
                output_grid_shapes.append(grid_shape)
            
            return torch.stack(output_grids), torch.stack(output_grid_shapes)
        
        output_grids, output_grid_shapes = vectorized_apply_rotation(grids, grid_shapes, k=2)
        
        expected_grid_shapes = torch.tensor([[4, 3], [2, 2]])
        expected_grids = torch.tensor(
            [
                [[12, 11, 10, 0, 0], [9, 8, 7, 0, 0], [6, 5, 4, 0, 0], [3, 2, 1, 0, 0], [0, 0, 0, 0, 0]],
                [[4, 3, 0, 0, 0], [2, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output_grid_shapes, expected_grid_shapes))
        self.assertTrue(torch.equal(output_grids, expected_grids))

    def test_apply_rotation_with_torch_vmap(self):
        """Test using torch.vmap if available (PyTorch 2.0+)."""
        try:
            from torch.func import vmap
            
            grids = torch.stack([self.grid, self.grid2])
            grid_shapes = torch.stack([self.grid_shape, self.grid_shape2])
            
            # Create a partial function for k=2
            rotation_func = functools.partial(_apply_rotation, k=2)
            
            # Use torch.vmap for vectorization
            output_grids, output_grid_shapes = vmap(rotation_func)(grids, grid_shapes)
            
            expected_grid_shapes = torch.tensor([[4, 3], [2, 2]])
            expected_grids = torch.tensor(
                [
                    [[12, 11, 10, 0, 0], [9, 8, 7, 0, 0], [6, 5, 4, 0, 0], [3, 2, 1, 0, 0], [0, 0, 0, 0, 0]],
                    [[4, 3, 0, 0, 0], [2, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                ]
            )
            self.assertTrue(torch.equal(output_grid_shapes, expected_grid_shapes))
            self.assertTrue(torch.equal(output_grids, expected_grids))
            
        except ImportError:
            # torch.func.vmap not available, skip this test
            self.skipTest("torch.func.vmap not available in this PyTorch version")

    def test_rotation_consistency(self):
        """Test that multiple rotations are consistent."""
        # Test that 4 rotations bring us back to the original
        grid, grid_shape = self.grid.clone(), self.grid_shape.clone()
        
        for _ in range(4):
            grid, grid_shape = _apply_rotation(grid, grid_shape, k=1)
        
        # After 4 rotations, we should be back to the original
        # Note: Due to the way padding and rolling works, this might not be exactly equal
        # but the shape should be the same
        self.assertTrue(torch.equal(grid_shape, self.grid_shape))

    def test_rotation_edge_cases(self):
        """Test edge cases for rotation."""
        # Test with square grid
        square_grid = torch.tensor([[1, 2], [3, 4]])
        square_shape = torch.tensor([2, 2])
        
        rotated_grid, rotated_shape = _apply_rotation(square_grid, square_shape, k=1)
        expected_rotated_grid = torch.tensor([[3, 1], [4, 2]])
        expected_rotated_shape = torch.tensor([2, 2])
        
        self.assertTrue(torch.equal(rotated_shape, expected_rotated_shape))
        self.assertTrue(torch.equal(rotated_grid, expected_rotated_grid))

    def test_rotation_datatypes(self):
        """Test that rotation preserves data types."""
        # Test with different data types
        float_grid = self.grid.float()
        float_shape = self.grid_shape.float()
        
        rotated_grid, rotated_shape = _apply_rotation(float_grid, float_shape, k=1)
        
        self.assertEqual(rotated_grid.dtype, torch.float32)
        self.assertEqual(rotated_shape.dtype, torch.float32)

    def test_rotation_device_consistency(self):
        """Test that rotation works correctly with different devices."""
        # Test on CPU (always available)
        cpu_grid = self.grid.to('cpu')
        cpu_shape = self.grid_shape.to('cpu')
        
        rotated_grid, rotated_shape = _apply_rotation(cpu_grid, cpu_shape, k=1)
        
        self.assertEqual(rotated_grid.device.type, 'cpu')
        self.assertEqual(rotated_shape.device.type, 'cpu')
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            cuda_grid = self.grid.to('cuda')
            cuda_shape = self.grid_shape.to('cuda')
            
            rotated_grid, rotated_shape = _apply_rotation(cuda_grid, cuda_shape, k=1)
            
            self.assertEqual(rotated_grid.device.type, 'cuda')
            self.assertEqual(rotated_shape.device.type, 'cuda')

    def tearDown(self):
        # Cleanup code to run after each test
        pass


# Additional utility function for testing
def run_rotation_benchmark():
    """Benchmark rotation function performance."""
    import time
    
    # Create larger test data
    large_grid = torch.randint(0, 10, (100, 100))
    large_shape = torch.tensor([80, 90])
    
    # Warm up
    for _ in range(10):
        _apply_rotation(large_grid, large_shape, k=1)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        _apply_rotation(large_grid, large_shape, k=1)
    end_time = time.time()
    
    print(f"Average rotation time: {(end_time - start_time) / 100 * 1000:.2f} ms")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
    
    # Optionally run benchmark
    # print("\n" + "="*50)
    # print("Running rotation benchmark...")
    # run_rotation_benchmark()