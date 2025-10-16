from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.vit import ActorCritic_ViT
from utils.obs_stats import ObsStats


@dataclass
class PolicyValueConfig:
    """Configuration for the policy-value network wrapper."""

    action_size: int = 9000
    normalize_obs: bool = True


class PolicyValueNet(nn.Module):
    """Wrap the ViT backbone to deliver masked policy logits and scalar values."""

    def __init__(self, cfg: PolicyValueConfig, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.backbone = ActorCritic_ViT().to(self.device)
        self.obs_stats = ObsStats((30, 180)) if cfg.normalize_obs else None

    def forward(self, obs_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.backbone(obs_tensor)
        return logits, value.squeeze(-1)

    @torch.no_grad()
    def infer_state(self, state) -> Tuple[np.ndarray, float]:
        policies, values = self.infer_states([state])
        return policies[0], float(values[0])

    @torch.no_grad()
    def infer_states(self, states: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        if not states:
            raise ValueError("State batch must be non-empty")

        obs_batch = []
        mask_batch = []
        for state in states:
            obs_batch.append(state.encode_obs())
            mask_batch.append(state.build_mask())

        obs_array = np.stack(obs_batch)
        if self.obs_stats is not None:
            obs_array = self.obs_stats.normalize_obs(obs_array)

        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(np.stack(mask_batch), dtype=torch.float32, device=self.device)

        was_training = self.backbone.training
        self.backbone.eval()
        logits, values = self.forward(obs_tensor)
        masked_logits = logits + mask_tensor
        policy = F.softmax(masked_logits, dim=-1)
        if was_training:
            self.backbone.train()

        return policy.cpu().numpy(), values.cpu().numpy()

    def update_norm(self, obs_batch: Iterable[np.ndarray]) -> None:
        if self.obs_stats is None:
            return
        stacked = np.stack(list(obs_batch))
        self.obs_stats.update_stats(stacked)
