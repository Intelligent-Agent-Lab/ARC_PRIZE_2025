import numpy as np
from dataclasses import dataclass


@dataclass
class ObsStats:
    """Maintain running statistics for observation normalization."""

    shape: tuple
    epsilon: float = 1e-8

    def __post_init__(self):
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = self.epsilon

    def update_stats(self, batch: np.ndarray) -> None:
        """Update running statistics from a batch of observations."""
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == len(self.shape):
            batch = batch[np.newaxis, ...]

        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        new_var = m2 / total

        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using accumulated statistics."""
        return (obs - self.mean) / np.sqrt(self.var + self.epsilon)
