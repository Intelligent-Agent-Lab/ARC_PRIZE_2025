from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from arc_agi_grid_env_coord import action_converter


@dataclass
class ArcGameState:
    """Immutable snapshot of an ARC puzzle for search."""

    grid: np.ndarray
    target: np.ndarray
    size_candidate: Sequence[int]
    color_candidate: Sequence[int]
    step: int
    done: bool
    outcome: float | None

    @classmethod
    def from_env(cls, obs: np.ndarray, info: dict, step: int = 0) -> "ArcGameState":
        target = np.array(info["target_grid_img"], copy=True)
        size_candidate = info.get("size_candidate", [30, 30])
        color_candidate = info.get("color_candidate", list(range(10)))
        grid = np.array(obs, copy=True)
        done, outcome = cls._check_terminal(grid, target)
        return cls(grid, target, tuple(size_candidate), tuple(color_candidate), step, done, outcome)

    @staticmethod
    def _check_terminal(grid: np.ndarray, target: np.ndarray) -> tuple[bool, float | None]:
        if np.array_equal(grid, target):
            return True, 1.0
        solution = grid[:, 150:]
        target_area = target[:, 150:]
        if not np.any(solution == 11):
            return True, -1.0
        if np.any((solution == 11) & (target_area != 11)):
            return False, None
        return False, None

    def clone_state(self) -> "ArcGameState":
        return ArcGameState(
            np.array(self.grid, copy=True),
            np.array(self.target, copy=True),
            tuple(self.size_candidate),
            tuple(self.color_candidate),
            self.step,
            self.done,
            self.outcome,
        )

    def encode_obs(self) -> np.ndarray:
        return np.array(self.grid, copy=True)

    def build_mask(self) -> np.ndarray:
        if self.done:
            return np.full(9000, -1e9, dtype=np.float32)

        mask = np.zeros(9000, dtype=np.float32)
        solution = self.grid[:, 150:]
        target_rows, target_cols = self._parse_size()
        valid_colors = {int(c) for c in self.color_candidate} if self.color_candidate else set(range(10))

        valid_positions = np.zeros((30, 30), dtype=bool)
        max_rows = min(target_rows, 30)
        max_cols = min(target_cols, 30)
        if max_rows > 0 and max_cols > 0:
            valid_positions[:max_rows, :max_cols] = solution[:max_rows, :max_cols] == 11
        else:
            valid_positions = solution == 11

        row_idx = np.arange(30).reshape(-1, 1)
        col_idx = np.arange(30).reshape(1, -1)
        indices = row_idx * 30 + col_idx

        for color in range(10):
            offset = color * 900
            flat_idx = offset + indices
            if color not in valid_colors:
                mask[flat_idx] = -1e9
            else:
                invalid = ~valid_positions
                if np.any(invalid):
                    mask[flat_idx[invalid]] = -1e9

        return mask

    def list_moves(self) -> List[int]:
        if self.done:
            return []
        mask = self.build_mask()
        return np.where(mask == 0)[0].tolist()

    def step_action(self, action: int) -> "ArcGameState":
        if self.done:
            return self

        parsed = action_converter(action)
        color = int(parsed["color"])
        row, col = map(int, parsed["coordinate"])
        sol_col = 150 + col

        new_grid = np.array(self.grid, copy=True)
        current = new_grid[row, sol_col]
        target_color = int(self.target[row, sol_col])
        new_grid[row, sol_col] = color

        if color != target_color or current != 11:
            return ArcGameState(new_grid, self.target, self.size_candidate, self.color_candidate, self.step + 1, True, -1.0)

        solved = np.array_equal(new_grid, self.target)
        if solved:
            return ArcGameState(new_grid, self.target, self.size_candidate, self.color_candidate, self.step + 1, True, 1.0)

        return ArcGameState(new_grid, self.target, self.size_candidate, self.color_candidate, self.step + 1, False, None)

    def is_terminal(self) -> bool:
        return self.done

    def get_outcome(self) -> float | None:
        return self.outcome

    def _parse_size(self) -> tuple[int, int]:
        if not self.size_candidate:
            return 30, 30
        if len(self.size_candidate) >= 2:
            return int(self.size_candidate[0]), int(self.size_candidate[1])
        side = int(self.size_candidate[0])
        return side, side
