from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from arc_agi_grid_env_coord import action_converter
from mcts.game_state import ArcGameState
from mcts.mcts import MCTS


@dataclass
class EpisodeStats:
    moves: int
    outcome: float
    reward: float


class SelfPlayRunner:
    def __init__(self, env, mcts: MCTS, max_moves: int = 64):
        self.env = env
        self.mcts = mcts
        self.max_moves = max_moves

    def run_episode(self, reset_options: Optional[Dict] = None) -> Tuple[List[np.ndarray], List[np.ndarray], EpisodeStats]:
        obs, info = self.env.reset(options=reset_options)
        state = ArcGameState.from_env(obs, info)
        obs_history: List[np.ndarray] = []
        policy_history: List[np.ndarray] = []
        reward = 0.0

        for move in range(self.max_moves):
            search_policy, _ = self.mcts.run_search(state)
            policy, action = self._select_action(state, search_policy)
            obs_history.append(state.encode_obs())
            policy_history.append(policy)

            env_action = action_converter(action)
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated
            state = ArcGameState.from_env(obs, info, step=move + 1)
            if done:
                break

        outcome = 1.0 if reward > 0 else -1.0 if reward < 0 else 0.0
        stats = EpisodeStats(moves=len(obs_history), outcome=outcome, reward=reward)
        return obs_history, policy_history, stats

    def _select_action(self, state: ArcGameState, policy: np.ndarray) -> Tuple[np.ndarray, int]:
        if policy.sum() <= 0:
            moves = state.list_moves()
            if not moves:
                fallback = np.zeros_like(policy)
                fallback[0] = 1.0
                return fallback, 0
            probs = np.ones(len(moves), dtype=np.float32) / len(moves)
            action = int(np.random.choice(moves, p=probs))
            dist = np.zeros_like(policy)
            for move, prob in zip(moves, probs):
                dist[move] = prob
            return dist, action
        action = int(np.random.choice(np.arange(policy.shape[0]), p=policy))
        return policy, action
