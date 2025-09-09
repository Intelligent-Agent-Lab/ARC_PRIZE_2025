import os
import sys
import random
import time
import math
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from arc_agi_grid_env_coord import cmap
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ale_py

# Import our modules
from matplotlib import colors
from pathlib import Path

from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.TimeLimit(env, 500)
        # env = gym.wrappers.Autoreset(env, )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        return env

    return thunk


def make_atari_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class CNNAgent(nn.Module):
    """Vision Transformer based Actor-Critic network for PPO with grid observations."""
    
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
       
    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class ArcAgiVectorizedTrainer:
    """Trainer class for PPO with vectorized ArcAgiGrid environments."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_environment()
        self.setup_agent()
        self.setup_logging()
        self.setup_storage()
        
    def setup_environment(self):
        """Setup the vectorized training environments."""
        # Load data

        self.seed = self.config.environment.seed
        self.num_envs = self.config.environment.num_envs
        self.num_steps = self.config.environment.num_steps
        
        # Create vectorized environment
        run_name = f"{self.config.environment.env_id}__{self.config.environment.seed}__{int(time.time())}"
        if 'ALE' in self.config.environment.env_id:
            self.envs = gym.vector.AsyncVectorEnv(
            [make_atari_env(self.config.environment.env_id, i, self.config.environment.capture_video, run_name) for i in range(self.config.environment.num_envs)],
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
            )
        else:
            self.envs = gym.vector.AsyncVectorEnv(
                [make_env(self.config.environment.env_id, i, self.config.environment.capture_video, run_name) for i in range(self.config.environment.num_envs)],
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
            )
            
        print(f"Vectorized environments created successfully!")
        print(f"Number of environments: {self.num_envs}")
        print(f"Single observation space: {self.envs.single_observation_space}")
        print(f"Single action space: {self.envs.single_action_space}")
        
    def setup_agent(self):
        """Setup the PPO agent."""
        # Calculate action space size
        print(f"single_obs_space: {self.envs.single_observation_space}")
        print(f"single_action_space: {self.envs.single_action_space}")
        
        obs_size = np.prod(self.envs.single_observation_space.shape)
        action_size = self.envs.single_action_space.n 
        
        print(f"setup_agent. obs_size: {obs_size}, action_size: {action_size}")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # Create agent
        if 'ALE' in self.config.environment.env_id:
            self.agent = CNNAgent(self.envs).to(self.device)
        else:
            self.agent = MLPAgent(self.envs).to(self.device)
            
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(), 
            lr=self.config.training.learning_rate, 
            eps=1e-5
        )
        batch_size = int(self.config.environment.num_envs * self.config.environment.num_steps)
        self.mini_batch_size = int(batch_size // self.config.training.num_minibatches)
        print(f"PPO Agent created with device: {self.device}")
        
    def setup_logging(self):
        """Setup logging and metrics tracking."""
        self.episode_returns = deque(maxlen=self.config.environment.num_envs)
        self.episode_lengths = deque(maxlen=self.config.environment.num_envs)
        self.success_rate = deque(maxlen=self.config.environment.num_envs)
        
        # Create save directory
        os.makedirs(self.config.logging.save_dir, exist_ok=True)
        
        # Initialize wandb logger
        if self.config.logging.use_wandb:
            run_name = f"ppo_vectorized_{self.config.environment.env_id}_{int(time.time())}"
            wandb.init(
                project=self.config.logging.wandb_project,
                config=OmegaConf.to_container(self.config, resolve=True),
                name=run_name,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True
            )
        
        # Initialize tensorboard logger
        if self.config.logging.use_tensorboard:
            run_name = f"ppo_vectorized_{self.config.environment.env_id}_{int(time.time())}"
            self.tensorboard_writer = SummaryWriter(f"runs/{run_name}")
            self.tensorboard_writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(self.config, resolve=True).items()])),
            )
        else:
            self.tensorboard_writer = None
    
    def setup_storage(self):
        """Setup storage for rollout data."""
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        
        # Manual episode tracking
        self.current_episode_returns = np.zeros(self.num_envs)
        self.current_episode_lengths = np.zeros(self.num_envs)
        
    def collect_rollouts(self, next_obs, next_done, iteration: int):
        """Collect rollout data for training using vectorized environments."""
        # Reset environments

        # Reset episode tracking
        for step in range(self.num_steps):
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # Execute actions in vectorized environment
            next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            # for i in range(self.num_envs):
                # if self.current_episode_returns[i] >= 500.0:
                    # truncations[i] = True
            # print(f'rollout step: {step}. reward: {reward}')
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
            
            # Update episode tracking
            self.current_episode_returns += reward
            self.current_episode_lengths += 1
            
            if 'episode' in infos.keys():
                print(f'f"iteration={iteration}, episodic_return={infos['episode']['r']}')
            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             print(f"iteration: {iteration}: episodic_return={info['episode']['r']}")

            # Log episode statistics when episodes end
            # print(f'rollout step: {step} before reset. current_episode_returns: {self.current_episode_returns}',)
            # print(f'terminations: {terminations}. truncations: {truncations}',)
            
            
            for i in range(self.num_envs):
                if next_done[i]:
                    # print(f"env {i} is done")
                    # Episode ended - log stats
                    self.episode_returns.append(self.current_episode_returns[i])
                    self.episode_lengths.append(self.current_episode_lengths[i])
                    # self.success_rate.append(1.0 if self.current_episode_returns[i] > 400.0 else 0.0)
                    
                    # Reset tracking for this environment
                    self.current_episode_returns[i] = 0.0
                    self.current_episode_lengths[i] = 0
            # print(f'rollout step: {step} after reset. current_episode_returns: {self.current_episode_returns}')
            # print(f'terminations: {terminations}. truncations: {truncations}',)
            
            # print(f'rollout step: {step}. current_episode_returns: {self.current_episode_returns}')
        # Bootstrap value for the next state
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            
        return next_value

    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        with torch.no_grad():
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.config.training.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.config.training.gamma * self.config.training.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values
            
        return advantages, returns

    def update_agent(self, advantages, returns, global_step):
        """Update the agent using PPO."""
        # Flatten the batch
        batch_size = self.num_envs * self.num_steps
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for epoch in range(self.config.training.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.training.eps_clip).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.config.training.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.training.eps_clip, 1 + self.config.training.eps_clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.training.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.training.eps_clip,
                        self.config.training.eps_clip,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.config.training.entropy_coef * entropy_loss + v_loss * self.config.training.value_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()

            if self.config.training.target_kl is not None and approx_kl > self.config.training.target_kl:
                break

        # Calculate explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            'policy_loss': pg_loss.item(),
            'value_loss': v_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'old_approx_kl': old_approx_kl.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': np.mean(clipfracs),
            'explained_variance': explained_var
        }

    def train(self):
        """Main training loop."""
        print("Starting PPO vectorized training...")
        print(f"Configuration: {self.config}")
        
        # Calculate training parameters
        batch_size = self.num_envs * self.num_steps
        num_iterations = self.config.training.total_timesteps // batch_size
        
        print(f"Batch size: {batch_size}")
        print(f"Number of iterations: {num_iterations}")
        
        global_step = 0
        start_time = time.time()
        best_mean_reward = -float('inf')
        first_vis = True
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        for iteration in range(1, num_iterations + 1):
            # Annealing learning rate
            if self.config.training.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * self.config.training.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            # Collect rollouts
            next_value = self.collect_rollouts(next_obs, next_done, iteration)
            
            # Compute GAE
            advantages, returns = self.compute_gae(next_value)
            
            # Update global step
            global_step += batch_size
            
            # Update agent
            training_metrics = self.update_agent(advantages, returns, global_step)
            
            # Logging
            if iteration % self.config.logging.log_interval == 0:
                mean_reward = np.mean(self.episode_returns) if self.episode_returns else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                success_rate = np.mean(self.success_rate) if self.success_rate else 0
                sps = int(global_step / (time.time() - start_time))
                
                print(f"\nIteration {iteration}/{num_iterations}")
                print(f"Global step: {global_step}")
                # print(f"Last return: {self.episode_returns[-1]:.3f}")
                print(f"Mean return (last {self.num_envs} episodes): {mean_reward:.3f}")
                print(f"Mean episode length: {mean_length:.1f}")
                print(f"Success rate: {success_rate:.3f}")
                print(f"SPS: {sps}")
                print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                if training_metrics:
                    print(f"Policy loss: {training_metrics['policy_loss']:.4f}")
                    print(f"Value loss: {training_metrics['value_loss']:.4f}")
                    print(f"Entropy loss: {training_metrics['entropy_loss']:.4f}")
                    print(f"Explained variance: {training_metrics['explained_variance']:.4f}")
                
                # Log to wandb
                if self.config.logging.use_wandb:
                    log_dict = {
                        'charts/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'charts/last_episodic_return': self.episode_returns[-1],
                        'charts/mean_episodic_return': mean_reward,
                        'charts/episodic_length': mean_length,
                        'charts/SPS': sps,
                        'losses/policy_loss': training_metrics['policy_loss'],
                        'losses/value_loss': training_metrics['value_loss'],
                        'losses/entropy': training_metrics['entropy_loss'],
                        'losses/old_approx_kl': training_metrics['old_approx_kl'],
                        'losses/approx_kl': training_metrics['approx_kl'],
                        'losses/clipfrac': training_metrics['clipfrac'],
                        'losses/explained_variance': training_metrics['explained_variance']
                    }
                    wandb.log(log_dict, step=global_step)
                
                # Log to tensorboard
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar('charts/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
                    self.tensorboard_writer.add_scalar('charts/episodic_return', mean_reward, global_step)
                    self.tensorboard_writer.add_scalar('charts/episodic_length', mean_length, global_step)
                    self.tensorboard_writer.add_scalar('charts/SPS', sps, global_step)
                    self.tensorboard_writer.add_scalar('losses/policy_loss', training_metrics['policy_loss'], global_step)
                    self.tensorboard_writer.add_scalar('losses/value_loss', training_metrics['value_loss'], global_step)
                    self.tensorboard_writer.add_scalar('losses/entropy', training_metrics['entropy_loss'], global_step)
                    self.tensorboard_writer.add_scalar('losses/old_approx_kl', training_metrics['old_approx_kl'], global_step)
                    self.tensorboard_writer.add_scalar('losses/approx_kl', training_metrics['approx_kl'], global_step)
                    self.tensorboard_writer.add_scalar('losses/clipfrac', training_metrics['clipfrac'], global_step)
                    self.tensorboard_writer.add_scalar('losses/explained_variance', training_metrics['explained_variance'], global_step)
            
            # Save checkpoint
            if iteration % self.config.logging.save_interval == 0 and iteration > 0:
                checkpoint_path = os.path.join(self.config.logging.save_dir, f'checkpoint_{iteration}.pth')
                torch.save({
                    'agent_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'iteration': iteration,
                    'global_step': global_step,
                    'config': self.config
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
                
                # Save best model if performance improved
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_path = os.path.join(self.config.logging.save_dir, 'best_model.pth')
                    torch.save({
                        'agent_state_dict': self.agent.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'iteration': iteration,
                        'global_step': global_step,
                        'config': self.config,
                        'best_reward': best_mean_reward
                    }, best_path)
                    print(f"New best model saved! Mean reward: {best_mean_reward:.3f}")
        
        print("Training completed!")
        
        # Final save
        final_path = os.path.join(self.config.logging.save_dir, 'final_model.pth')
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': num_iterations,
            'global_step': global_step,
            'config': self.config
        }, final_path)
        print(f"Final model saved: {final_path}")
        
        # Close loggers
        if self.config.logging.use_wandb:
            wandb.finish()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        self.envs.close()


@hydra.main(version_base=None, config_path="config", config_name="ppo_breakout")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    print("Vectorized PPO Training configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seeds for reproducibility
    random.seed(cfg.environment.seed)
    np.random.seed(cfg.environment.seed)
    torch.manual_seed(cfg.environment.seed)
    torch.backends.cudnn.deterministic = True
    
    # Create trainer and start training
    trainer = ArcAgiVectorizedTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
