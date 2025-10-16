from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass
class ReplaySample:
    obs: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: Deque[ReplaySample] = deque(maxlen=capacity)

    def add_sample(self, obs: np.ndarray, policy: np.ndarray, value: float) -> None:
        self.storage.append(ReplaySample(obs.copy(), policy.copy(), float(value)))

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.storage:
            raise ValueError("Replay buffer is empty")
        count = len(self.storage)
        indices = np.random.choice(count, size=min(batch_size, count), replace=False)
        obs_batch = np.stack([self.storage[idx].obs for idx in indices])
        policy_batch = np.stack([self.storage[idx].policy for idx in indices])
        value_batch = np.array([self.storage[idx].value for idx in indices], dtype=np.float32)
        return obs_batch, policy_batch, value_batch

    def buffer_size(self) -> int:
        return len(self.storage)
