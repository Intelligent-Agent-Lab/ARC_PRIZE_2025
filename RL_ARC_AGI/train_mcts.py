import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from arc_agi_grid_env_coord import (
    create_arc_env_coord,
    load_challenges_and_solutions,
    preprocess_data,
)
from mcts.mcts import MCTS, MCTSConfig
from mcts.replay_buffer import ReplayBuffer
from mcts.self_play import SelfPlayRunner
from network.policy_value import PolicyValueConfig, PolicyValueNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(cfg: DictConfig):
    training_challenges, training_solutions, evaluation_challenges, evaluation_solutions, test_challenges = (
        load_challenges_and_solutions(
            cfg.environment.training_challenges_json,
            cfg.environment.training_solutions_json,
            cfg.environment.evaluation_challenges_json,
            cfg.environment.evaluation_solutions_json,
            cfg.environment.test_challenges_json,
        )
    )

    train_task_img_dict, _, _, _ = preprocess_data(training_challenges, training_solutions)
    eval_task_img_dict, _, _, _ = preprocess_data(evaluation_challenges, evaluation_solutions)

    env = create_arc_env_coord(
        fixed_task=cfg.environment.fixed_task,
        fixed_pair_idx=cfg.environment.fixed_pair_idx,
        task_id=cfg.environment.task_id,
        pair_idx=cfg.environment.pair_idx,
        training_challenges=training_challenges,
        training_solutions=training_solutions,
        evaluation_challenges=evaluation_challenges,
        evaluation_solutions=evaluation_solutions,
        test_challenges=test_challenges,
        train_task_img_dict=train_task_img_dict,
        eval_task_img_dict=eval_task_img_dict,
    )
    return env


def train_loop(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    env = make_env(cfg)
    net_cfg = PolicyValueConfig(normalize_obs=cfg.network.normalize_obs)
    net = PolicyValueNet(net_cfg, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    mcts_cfg = MCTSConfig(
        simulations=cfg.mcts.simulations,
        cpuct=cfg.mcts.cpuct,
        dirichlet_alpha=cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon=cfg.mcts.dirichlet_epsilon,
        temperature=cfg.mcts.temperature,
        min_temperature=cfg.mcts.min_temperature,
        temperature_decay=cfg.mcts.temperature_decay,
    )
    planner = MCTS(net, mcts_cfg)
    runner = SelfPlayRunner(env, planner, max_moves=cfg.self_play.max_moves)
    buffer = ReplayBuffer(cfg.training.buffer_size)

    reset_options = {
        "mode": cfg.environment.mode,
        "task_id": cfg.environment.task_id if cfg.environment.fixed_task else None,
        "pair_idx": cfg.environment.pair_idx if cfg.environment.fixed_pair_idx else None,
        "reset_sol_grid": cfg.environment.reset_sol_grid,
        "rand_init": cfg.environment.rand_init,
    }

    for iteration in range(cfg.training.num_iterations):
        total_moves = 0
        wins = 0
        losses = 0

        for _ in range(cfg.self_play.episodes_per_iter):
            obs_history, policy_history, stats = runner.run_episode(reset_options=reset_options)
            if cfg.network.normalize_obs and obs_history:
                net.update_norm(obs_history)
            for obs, policy in zip(obs_history, policy_history):
                buffer.add_sample(obs, policy, stats.outcome)
            total_moves += stats.moves
            if stats.outcome > 0:
                wins += 1
            elif stats.outcome < 0:
                losses += 1

        if buffer.buffer_size() < cfg.training.batch_size:
            continue

        net.train()
        for _ in range(cfg.training.updates_per_iter):
            obs_batch, policy_batch, value_batch = buffer.sample_batch(cfg.training.batch_size)
            if cfg.network.normalize_obs:
                obs_batch = net.obs_stats.normalize_obs(obs_batch)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            policy_tensor = torch.tensor(policy_batch, dtype=torch.float32, device=device)
            value_tensor = torch.tensor(value_batch, dtype=torch.float32, device=device)

            logits, values = net.forward(obs_tensor)
            log_probs = torch.log_softmax(logits, dim=-1)
            policy_loss = -(policy_tensor * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(values, value_tensor)
            entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()

            loss = (
                cfg.training.policy_coef * policy_loss
                + cfg.training.value_coef * value_loss
                - cfg.training.entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.training.max_grad_norm)
            optimizer.step()

        avg_moves = total_moves / max(cfg.self_play.episodes_per_iter, 1)
        print(
            f"Iter {iteration:04d} | episodes={cfg.self_play.episodes_per_iter} | "
            f"avg_moves={avg_moves:.1f} | wins={wins} | losses={losses} | buffer={buffer.buffer_size()}"
        )

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ckpt_dir / "policy_value_net.pt")
    print(f"Saved checkpoint to {ckpt_dir}")


@hydra.main(config_path="config", config_name="mcts", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train_loop(cfg)


if __name__ == "__main__":
    main()
