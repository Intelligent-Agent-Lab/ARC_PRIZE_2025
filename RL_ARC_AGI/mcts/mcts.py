import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from mcts.game_state import ArcGameState
from network.policy_value import PolicyValueNet


@dataclass
class MCTSConfig:
    simulations: int = 64
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    min_temperature: float = 0.1
    temperature_decay: float = 0.99


class PUCTNode:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "PUCTNode"] = {}

    def expand_node(self, actions: Iterable[int], priors: np.ndarray) -> None:
        for action in actions:
            if action not in self.children:
                self.children[action] = PUCTNode(float(priors[action]))

    def select_child(self, parent_visit: int, cpuct: float) -> Tuple[int, "PUCTNode"]:
        best_score = -float("inf")
        best_action = -1
        best_child = None

        total = math.sqrt(parent_visit + 1e-8)
        for action, child in self.children.items():
            q_value = 0.0 if child.visit_count == 0 else child.value_sum / child.visit_count
            u_value = cpuct * child.prior * total / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("No child selected during MCTS traversal")
        return best_action, best_child

    def backup_value(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    def __init__(self, net: PolicyValueNet, cfg: MCTSConfig):
        self.net = net
        self.cfg = cfg
        self.temperature = cfg.temperature

    def run_search(self, root_state: ArcGameState) -> Tuple[np.ndarray, float]:
        if root_state.is_terminal():
            outcome = root_state.get_outcome()
            value = 0.0 if outcome is None else outcome
            return np.zeros(9000, dtype=np.float32), value

        root_policy, root_value = self.net.infer_state(root_state)
        root = PUCTNode(0.0)
        valid_actions = root_state.list_moves()
        if valid_actions:
            root.expand_node(valid_actions, root_policy)
            self._add_root_noise(root)

        for _ in range(self.cfg.simulations):
            node = root
            state = root_state.clone_state()
            path = [node]

            while node.children:
                action, next_node = node.select_child(node.visit_count, self.cfg.cpuct)
                state = state.step_action(action)
                path.append(next_node)
                node = next_node
                if state.is_terminal():
                    break

            if state.is_terminal():
                value = state.get_outcome()
                if value is None:
                    value = 0.0
            else:
                policy, value = self.net.infer_state(state)
                valid = state.list_moves()
                node.expand_node(valid, policy)

            for visited in reversed(path):
                visited.backup_value(value)

        policy_target = np.zeros(9000, dtype=np.float32)
        if root.children:
            visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
            actions = list(root.children.keys())
            if self.temperature <= 1e-6:
                best_idx = int(np.argmax(visits))
                policy_target[actions[best_idx]] = 1.0
            else:
                scaled = np.power(visits, 1.0 / self.temperature)
                scaled_sum = np.sum(scaled)
                if scaled_sum > 0:
                    probs = scaled / scaled_sum
                    for action, prob in zip(actions, probs):
                        policy_target[action] = prob

        self.temperature = max(self.cfg.min_temperature, self.temperature * self.cfg.temperature_decay)
        return policy_target, root_value

    def _add_root_noise(self, root: PUCTNode) -> None:
        if self.cfg.dirichlet_epsilon <= 0 or self.cfg.dirichlet_alpha <= 0:
            return
        items = list(root.children.items())
        if not items:
            return
        noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(items))
        for (action, child), eta in zip(items, noise):
            child.prior = child.prior * (1 - self.cfg.dirichlet_epsilon) + eta * self.cfg.dirichlet_epsilon
