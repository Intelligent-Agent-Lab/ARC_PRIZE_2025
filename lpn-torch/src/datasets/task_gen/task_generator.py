import functools
import random
from typing import Any, Optional

import networkx as nx
import numpy as np
import torch
from torch.utils.data import IterableDataset

from src.datasets.task_gen.utils import is_grid, run_with_timeout
from src.datasets.task_gen.re_arc_generators import GENERATORS_SRC_CODE, ARC_TASK_NAMES


class PatternTaskGenerator(IterableDataset):
    def __init__(
        self,
        num_pairs: int,
        seed: Optional[int] = None,
        num_rows: int = 10,
        num_cols: int = 10,
        pattern_size: int = 4,
        pattern_density: float = 1.0,
    ):
        self.num_pairs = num_pairs
        self.seed = seed
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.pattern_size = pattern_size
        self.pattern_density = pattern_density
        assert 1 <= self.pattern_size < min(self.num_rows, self.num_cols)
        assert 0.0 < self.pattern_density <= 1.0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id if self.seed is not None else None
        else:
            worker_seed = self.seed
        if worker_seed is not None:
            random.seed(worker_seed)
        return self

    def __next__(self) -> tuple[list[dict[str, tuple]], dict[str, Any]]:
        task = []
        pattern = self.generate_pattern()
        for _ in range(self.num_pairs):
            pair = self.generate_pair(pattern)
            task.append(pair)
        info = {"num_attempts_generate_task": 1, "G": nx.MultiDiGraph()}
        return task, info

    def generate_pattern(self) -> np.ndarray:
        pattern = np.zeros((self.pattern_size, self.pattern_size), dtype=int)
        for i in range(self.pattern_size):
            for j in range(self.pattern_size):
                if random.random() < self.pattern_density:
                    pattern[i, j] = random.randint(1, 9)
        return pattern

    def generate_pair(self, pattern: np.ndarray) -> dict[str, np.ndarray]:
        input_grid = np.zeros((self.num_rows, self.num_cols), dtype=int)
        output_grid = np.zeros((self.num_rows, self.num_cols), dtype=int)
        pattern_loc_row = random.randint(0, self.num_rows - self.pattern_size)
        pattern_loc_col = random.randint(0, self.num_cols - self.pattern_size)
        input_grid[pattern_loc_row, pattern_loc_col] = 1
        output_grid[
            pattern_loc_row : pattern_loc_row + self.pattern_size,
            pattern_loc_col : pattern_loc_col + self.pattern_size,
        ] = pattern
        return {"input": input_grid, "output": output_grid}


class ArcTrainTaskGenerator(IterableDataset):
    def __init__(
        self,
        num_pairs: int,
        seed: Optional[int] = None,
        timeout_generate_pair: int = 5,
        overfit_task: Optional[str] = None,
        only_n_tasks: Optional[int] = None,
    ):
        self.num_pairs = num_pairs
        self.seed = seed
        self.timeout_generate_pair = timeout_generate_pair
        self.random_state = None
        self.generate_functions = []
        if overfit_task is not None and only_n_tasks is not None:
            raise ValueError("Cannot specify both overfit_task and only_n_tasks.")
        self.overfit_task = overfit_task
        self.only_n_tasks = only_n_tasks
        self.task_names = ARC_TASK_NAMES
        if only_n_tasks is not None:
            self.task_names = self.task_names[:only_n_tasks]

    def __iter__(self):
        exec(GENERATORS_SRC_CODE, globals())  # add the generate functions to the global namespace
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id if self.seed is not None else None
        else:
            worker_seed = self.seed
        if worker_seed is not None:
            random.seed(worker_seed)
        self.random_state = random.getstate()
        if self.overfit_task is not None:
            task_fn_name = f"generate_{self.overfit_task}"
            assert task_fn_name in globals(), f"Function {task_fn_name} not found."
            self.generate_functions = [functools.partial(globals()[task_fn_name], 0, 1)]
        else:
            self.generate_functions = [
                functools.partial(globals()[f"generate_{task_name}"], 0, 1) for task_name in self.task_names
            ]
        return self

    def __next__(self) -> tuple[list[dict[str, tuple]], dict[str, Any]]:
        stop = False
        num_attempts = 0
        while not stop:
            stop = True
            num_attempts += 1
            program_id = random.randint(0, len(self.generate_functions) - 1)
            generate_fn = self.generate_functions[program_id]
            task = []
            for _ in range(self.num_pairs):
                try:
                    if self.timeout_generate_pair:
                        # Use a signal to run the function with a timeout
                        pair, self.random_state, exception = run_with_timeout(
                            generate_fn, timeout=self.timeout_generate_pair
                        )(random_state=self.random_state)
                        if exception is not None:
                            raise exception
                    else:
                        # Run the function without a timeout
                        pair = generate_fn()
                except KeyboardInterrupt:
                    raise
                except Exception:
                    stop = False
                    break
                if not is_grid(pair["input"]) or not is_grid(pair["output"]):
                    stop = False
                    break
                task.append({key: np.array(value) for key, value in pair.items()})
        info = {"num_attempts_generate_task": num_attempts, "program_id": program_id}
        return task, info


if __name__ == "__main__":
    from src.datasets.task_gen.utils import plot_task

    task_gen = ArcTrainTaskGenerator(num_pairs=4, seed=None)
    task, info = next(iter(task_gen))
    print(f"Generated a valid task after {info['num_attempts_generate_task']} attempts.")
    plot_task(task, figsize_factor=2)
    if "program_id" in info:
        print(f"Program ID: {info['program_id']}")
