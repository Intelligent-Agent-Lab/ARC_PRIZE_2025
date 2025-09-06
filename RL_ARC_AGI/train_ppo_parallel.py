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
from network.mlp import ActorCritic_MLP
from network.vit import ActorCritic_ViT
from arc_agi_grid_env_coord import cmap
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from arc_agi_grid_env_coord import ArcAgiGridEnvCoord, create_arc_env_coord, \
                                    action_converter, vectorized_action_converter, \
                                    load_challenges_and_solutions, \
                                    preprocess_data
from matplotlib import colors
from pathlib import Path


def make_env(fixed_task: bool, 
             fixed_pair_idx: bool, 
             task_id: str, 
             pair_idx: int,
             training_challenges,
             training_solutions,
             evaluation_challenges,
             evaluation_solutions,
             test_challenges,
             train_task_img_dict,
             eval_task_img_dict):
    """Create and wrap environment for vectorized training."""
    def thunk():
        env = create_arc_env_coord(
                fixed_task=fixed_task, 
                fixed_pair_idx=fixed_pair_idx,
                task_id=task_id,
                pair_idx=pair_idx,
                training_challenges=training_challenges,
                training_solutions=training_solutions, 
                evaluation_challenges=evaluation_challenges,
                evaluation_solutions=evaluation_solutions,
                test_challenges=test_challenges,
                train_task_img_dict=train_task_img_dict,
                eval_task_img_dict=eval_task_img_dict,
            )
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


class Agent(nn.Module):
    """Vision Transformer based Actor-Critic network for PPO with grid observations."""
    
    def __init__(self, cfg,):
        super().__init__()
        if cfg.network.type == 'vit':
            self.ac_network = ActorCritic_ViT()
    
    def forward(self, x):
        """Forward pass returning CLS token representation."""
        # Patch embedding
        action_logits, state_value = self.ac_network(x)
        return action_logits, state_value 
       
    def get_value(self, x):
        _, state_value = self.forward(x)
        return state_value

    def get_action_and_value(self, x, action=None):
        action_logits, state_value = self.forward(x)
        probs = Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), state_value


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
        training_challenges_json = "../datasets/arc-agi_training_challenges.json"
        training_solutions_json = "../datasets/arc-agi_training_solutions.json" 
        evaluation_challenges_json = "../datasets/arc-agi_evaluation_challenges.json"
        evaluation_solutions_json = "../datasets/arc-agi_evaluation_solutions.json"
        test_challenges_json = "../datasets/arc-agi_test_challenges.json"

        training_challenges, training_solutions, evaluation_challenges, \
        evaluation_solutions, test_challenges = load_challenges_and_solutions(
                                                training_challenges_json,
                                                training_solutions_json,
                                                evaluation_challenges_json,
                                                evaluation_solutions_json,
                                                test_challenges_json,
                                            )
        train_task_img_dict, _ = preprocess_data(training_challenges, training_solutions)
        eval_task_img_dict, _ = preprocess_data(evaluation_challenges, evaluation_solutions)

        self.task_id_list = list(self.config.environment.task_id_list)
        self.seed = self.config.environment.seed
        self.fixed_task = self.config.environment.fixed_task
        self.fixed_pair_idx = self.config.environment.fixed_pair_idx
        self.task_id = self.config.environment.task_id
        self.pair_idx = self.config.environment.pair_idx
        self.num_envs = self.config.environment.num_envs
        self.num_steps = self.config.environment.num_steps
        
        # Create vectorized environment
        self.envs = gym.vector.SyncVectorEnv([
            make_env(
                fixed_task=self.fixed_task,
                fixed_pair_idx=self.fixed_pair_idx,
                task_id=self.task_id,
                pair_idx=self.pair_idx,
                training_challenges=training_challenges,
                training_solutions=training_solutions,
                evaluation_challenges=evaluation_challenges,
                evaluation_solutions=evaluation_solutions,
                test_challenges=test_challenges,
                train_task_img_dict=train_task_img_dict,
                eval_task_img_dict=eval_task_img_dict
            ) for i in range(self.num_envs)
        ])

        print(f"Vectorized environments created successfully!")
        print(f"Number of environments: {self.num_envs}")
        print(f"Single observation space: {self.envs.single_observation_space}")
        print(f"Single action space: {self.envs.single_action_space}")
        
    def setup_agent(self):
        """Setup the PPO agent."""
        # Calculate action space size
        obs_size = np.prod(self.envs.single_observation_space.shape)
        action_size = 9900 # 10 colors * 30x30 coordinate space = 9000
        
        print(f"setup_agent. obs_size: {obs_size}, action_size: {action_size}")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create agent
        self.agent = Agent(self.config).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(), 
            lr=self.config.training.learning_rate, 
            eps=1e-5
        )
        
        print(f"PPO Agent created with device: {self.device}")
        
    def setup_logging(self):
        """Setup logging and metrics tracking."""
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
        # Create save directory
        os.makedirs(self.config.logging.save_dir, exist_ok=True)
        
        # Initialize wandb logger
        if self.config.logging.use_wandb:
            run_name = f"ppo_vectorized_{self.task_id}_{int(time.time())}"
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
            run_name = f"ppo_vectorized_{self.task_id}_{int(time.time())}"
            self.tensorboard_writer = SummaryWriter(f"runs/{run_name}")
            self.tensorboard_writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(self.config, resolve=True).items()])),
            )
        else:
            self.tensorboard_writer = None
    
    def setup_storage(self):
        """Setup storage for rollout data."""
        self.obs = torch.zeros((self.num_steps, self.num_envs) + (30, 180)).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        
        # Manual episode tracking
        self.current_episode_returns = np.zeros(self.num_envs)
        self.current_episode_lengths = np.zeros(self.num_envs)
    
    def visualize_grid(self, iteration: int, w=0.5, first=False):
        """Visualize current grid state and log to wandb/tensorboard."""
        try:
            norm = colors.Normalize(vmin=0, vmax=11)
            # Get current grid from first environment
            info_dict = self.envs.envs[0]._get_info()
            target_grid = info_dict['target_grid_img']
            current_grid = info_dict['current_grid_img']
            timestep = info_dict['timestep']
            task_id = info_dict['task_id']
            test_input_idx = info_dict['test_input_idx']
            
            test_sol_current_mat = current_grid[:, 120:]
            test_sol_target_mat = target_grid[:, 120:]
                
            if first:
                plt.imshow(test_sol_target_mat, cmap=cmap, norm=norm)
                plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
                plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
                plt.grid(visible=True, which='both', color='#666666', linewidth=w)
                plt.xticks([x-0.5 for x in range(1 + len(test_sol_target_mat[0]))])
                plt.yticks([x-0.5 for x in range(1 + len(test_sol_target_mat))])
                plt.tick_params(axis='both', color='none', length=0)
                plt.title(f'task: #{task_id}  test_input_idx: #{test_input_idx}', fontsize=12, color='#000000')
                figure_folder_path = Path("./figures")
                if not figure_folder_path.exists():
                    figure_folder_path.mkdir(parents=True)
                plt.savefig(f"./figures/target.png")
                plt.close()

            plt.imshow(test_sol_current_mat, cmap=cmap, norm=norm)
            plt.grid(True, which='both', color='lightgrey', linewidth=1.0)
            plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
            plt.grid(visible=True, which='both', color='#666666', linewidth=w)
            plt.xticks([x-0.5 for x in range(1 + len(test_sol_current_mat[0]))])
            plt.yticks([x-0.5 for x in range(1 + len(test_sol_current_mat))])
            plt.tick_params(axis='both', color='none', length=0)
            plt.title(f'task: #{task_id}  test_input_idx: #{test_input_idx}  iter: #{iteration}  timestep: #{timestep}', 
                     fontsize=12, color='#000000')
            figure_folder_path = Path("./figures")
            if not figure_folder_path.exists():
                figure_folder_path.mkdir(parents=True)
            plt.savefig(f"./figures/{iteration}_current.png")
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not visualize grid at iteration {iteration}: {e}")
        
    def collect_rollouts(self, iteration: int):
        """Collect rollout data for training using vectorized environments."""
        # Reset environments
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        # Reset episode tracking
        self.current_episode_returns = np.zeros(self.num_envs)
        self.current_episode_lengths = np.zeros(self.num_envs)
        
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
            dict_action = vectorized_action_converter(action.cpu())
            # print(f'step: {step}. {dict_action}')
            next_obs, reward, terminations, truncations, infos = self.envs.step(dict_action)
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
            
            # Update episode tracking
            self.current_episode_returns += reward
            self.current_episode_lengths += 1
            
            # Log episode statistics when episodes end
            for i in range(self.num_envs):
                if terminations[i] or truncations[i]:
                    # Episode ended - log stats
                    self.episode_rewards.append(self.current_episode_returns[i])
                    self.episode_lengths.append(self.current_episode_lengths[i])
                    self.success_rate.append(1.0 if self.current_episode_returns[i] > 10.0 else 0.0)
                    
                    # Reset tracking for this environment
                    self.current_episode_returns[i] = 0.0
                    self.current_episode_lengths[i] = 0
        
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
            for start in range(0, batch_size, self.config.training.mini_batch_size):
                end = start + self.config.training.mini_batch_size
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
        
        for iteration in range(1, num_iterations + 1):
            # Annealing learning rate
            if self.config.training.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * self.config.training.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            # Collect rollouts
            next_value = self.collect_rollouts(iteration)
            
            # Compute GAE
            advantages, returns = self.compute_gae(next_value)
            
            # Update global step
            global_step += batch_size
            
            # Update agent
            training_metrics = self.update_agent(advantages, returns, global_step)
            
            # Logging
            if iteration % self.config.logging.log_interval == 0:
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                success_rate = np.mean(self.success_rate) if self.success_rate else 0
                sps = int(global_step / (time.time() - start_time))
                
                print(f"\nIteration {iteration}/{num_iterations}")
                print(f"Global step: {global_step}")
                print(f"Mean reward (last 100 episodes): {mean_reward:.3f}")
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
                        'charts/episodic_return': mean_reward,
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
            
            # Visualization
            if iteration % self.config.logging.visualize_period == 0:
                print(f"Creating grid visualization at iteration {iteration}...")
                self.visualize_grid(iteration, first=first_vis)
                first_vis = False
            
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


@hydra.main(version_base=None, config_path="config", config_name="ppo_vector_env")
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