import os
import sys
import random
import time
from typing import Dict, List, Any
import numpy as np
import torch
import gymnasium as gym
from collections import deque
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from arc_agi_grid_env import create_arc_env
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from arc_agi_grid_env import ArcAgiGridEnv
from env_wrappers import create_wrapped_env
from ppo_agent import PPOAgent
from matplotlib import colors
from pathlib import Path

class ArcAgiTrainer:
    """Trainer class for PPO on ArcAgiGrid environment."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_environment()
        self.setup_agent()
        self.setup_logging()
        
    def setup_environment(self):
        """Setup the training environment."""
        # Create base environment
        # base_env = ArcAgiGridEnv(
        #     training_challenges_json=self.config.environment.training_challenges_json,
        #     training_solutions_json=self.config.environment.training_solutions_json,
        #     evaluation_challenges_json=self.config.environment.evaluation_challenges_json,
        #     evaluation_solutions_json=self.config.environment.evaluation_solutions_json,
        #     test_challenges_json=self.config.environment.test_challenges_json
        # )
        
        # Wrap environment
        # self.env = create_wrapped_env(
        #     base_env, 
        #     normalize=self.config.environment.normalize_obs,
        #     reward_shaping=self.config.environment.reward_shaping
        # )
        self.env = create_arc_env(
            training_challenges_json=self.config.environment.training_challenges_json,
            training_solutions_json=self.config.environment.training_solutions_json,
            evaluation_challenges_json=self.config.environment.evaluation_challenges_json,
            evaluation_solutions_json=self.config.environment.evaluation_solutions_json,
            test_challenges_json=self.config.environment.test_challenges_json
            )
        self.task_id_list = list(self.config.environment.task_id_list)
        self.seed = self.config.environment.seed

        print(f"Environment created successfully!")
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")
        
    def setup_agent(self):
        """Setup the PPO agent."""
        obs_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        self.agent = PPOAgent(
            cfg=self.config,
            input_size=obs_size,
            action_size=action_size,
            hidden_size=self.config.network.hidden_size,
            learning_rate=self.config.training.learning_rate,
            gamma=self.config.training.gamma,
            eps_clip=self.config.training.eps_clip,
            value_coef=self.config.training.value_coef,
            entropy_coef=self.config.training.entropy_coef
        )
        
        print(f"PPO Agent created with device: {self.agent.device}")
        
    def setup_logging(self):
        """Setup logging and metrics tracking."""
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        self.training_metrics = []
        
        # Initialize wandb logger
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                config=OmegaConf.to_container(self.config, resolve=True),
                name=f"ppo_arc_agi_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Initialize tensorboard logger
        if self.config.logging.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(
                log_dir=os.path.join(self.config.logging.save_dir, "tensorboard_logs")
            )
        else:
            self.tensorboard_writer = None
    
    def visualize_grid(self, update: int, w=0.5, first=False):
        """Visualize current grid state and log to wandb/tensorboard."""
        cmap = colors.ListedColormap(
            ['#000000', # 0: black
            '#0074D9', # 1: blue
            '#FF4136', # 2: red
            '#2ECC40', # 3: green
            '#FFDC00', # 4: yello
            '#AAAAAA', # 5: gray
            '#F012BE', # 6: magenta
            '#FF851B', # 7: oragne
            '#7FDBFF', # 8: sky
            '#870C25', # 9: brwon
            '#FFFFFF', # 10: mask
            ])
        norm = colors.Normalize(vmin=0, vmax=10)
        # Get current grid from environment
        info_dict = self.env._get_info()  # Access the environment's current grid
        target_grid = info_dict['target_grid_img']
        current_grid = info_dict['current_grid_img']
        timestep = info_dict['timestep']
        task_id = info_dict['task_id']
        test_input_idx = info_dict['test_input_idx']
        
        test_sol_current_mat = current_grid[:, 120:]
        test_sol_target_mat = target_grid[:, 120:]
            
        target_path = Path(f"./figures/target.png")
        if first:
            plt.imshow(test_sol_target_mat, cmap=cmap, norm=norm)
            plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
            plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
            # '''Grid:'''
            plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
            plt.xticks([x-0.5 for x in range(1 + len(test_sol_target_mat[0]))])
            plt.yticks([x-0.5 for x in range(1 + len(test_sol_target_mat))])
            plt.tick_params(axis='both', color='none', length=0)
            '''sub title:'''
            plt.title(f'task: #{task_id}' + '  ' + f'test_input_idx: #{test_input_idx}  ', fontsize=12, color = '#000000')
            figure_folder_path = Path("./figures")
            if not figure_folder_path.exists():
                figure_folder_path.mkdir(parents=True)
            plt.savefig(f"./figures/target.png")
        try:

            # Create visualization
            plt.imshow(test_sol_current_mat, cmap=cmap, norm=norm)
            plt.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
            plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
            # '''Grid:'''
            plt.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
            plt.xticks([x-0.5 for x in range(1 + len(test_sol_current_mat[0]))])
            plt.yticks([x-0.5 for x in range(1 + len(test_sol_current_mat))])
            plt.tick_params(axis='both', color='none', length=0)
            '''sub title:'''
            plt.title(f'task: #{task_id}' + '  ' + f'test_input_idx: #{test_input_idx}  ' + f'update: #{update}  timestep: #{timestep}', fontsize=12, color = '#000000')
            figure_folder_path = Path("./figures")
            if not figure_folder_path.exists():
                figure_folder_path.mkdir(parents=True)
            plt.savefig(f"./figures/{update}_current.png")
            
            # Create visualization

            
            # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            # im = ax.imshow(current_grid, cmap='tab20', vmin=0, vmax=19)
            # ax.set_title(f"Agent Grid State - Update {update}")
            # ax.grid(True, color='white', linewidth=0.5)
            # ax.set_xticks(range(current_grid.shape[1]))
            # ax.set_yticks(range(current_grid.shape[0]))
            
            # # Add colorbar
            # plt.colorbar(im, ax=ax)
            
            # # Log to wandb
            # if self.config.logging.use_wandb:
            #     wandb.log({"grid_visualization": wandb.Image(fig), "update": update})
            
            # # Save and log to tensorboard
            # if self.tensorboard_writer:
            #     # Save figure to numpy array
            #     fig.canvas.draw()
            #     buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            #     buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #     self.tensorboard_writer.add_image("grid_visualization", buf, update, dataformats='HWC')
            plt.close()
            # plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not visualize grid at update {update}: {e}")
        
    def collect_rollouts(self, rollout_steps: int, update: int) -> Dict[str, float]:
        """Collect rollout data for training."""
        total_reward = 0
        total_steps = 0
        episodes_completed = 0
        successful_episodes = 0
        
        task_id = random.choice(self.task_id_list)
        
        obs, info = self.env.reset(
            seed=self.seed,
            options={
                'mode': 'train',
                'task_id': task_id,
                'reset_sol_grid': self.config.environment.reset_sol_grid
            }
        )
        
        for step in range(rollout_steps):
            # Select action
            action, log_prob, value = self.agent.select_action(obs)
            
            # Take environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            timestep = info['timestep']
            done = terminated or truncated
            print(f"update: {update}, timestep: {timestep}, action: {action}, terminated: {terminated}, truncated: {truncated}")
            # Store transition
            self.agent.store_transition(obs, action, log_prob, reward, value, done)
            
            total_reward += reward
            total_steps += 1
            obs = next_obs
            
            if done:
                episodes_completed += 1
                self.episode_rewards.append(total_reward)
                self.episode_lengths.append(total_steps)
                
                # Check if episode was successful (completed without early termination)
                if truncated and not terminated:
                    successful_episodes += 1
                    self.success_rate.append(1.0)
                else:
                    self.success_rate.append(0.0)
                # Reset environment
                obs, info = self.env.reset(
                    seed=self.seed,
                    options={
                        'mode': 'train',
                        'task_id': task_id,
                        'reset_sol_grid': self.config.environment.reset_sol_grid
                    }
                )
                total_reward = 0
                total_steps = 0
        
        # Get final value for bootstrap
        _, _, final_value = self.agent.select_action(obs)
        
        return {
            'total_reward': total_reward,
            'episodes_completed': episodes_completed,
            'successful_episodes': successful_episodes,
            'final_value': final_value
        }
    
    def evaluate_agent(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current agent performance."""
        eval_rewards = []
        eval_success = []
        
        task_id = random.choice(self.task_id_list)
        
        for episode in range(num_episodes):
            obs, info = self.env.reset(
                seed=self.seed,  # Fixed seed for reproducible evaluation
                options={
                    'mode': 'train',  # Can change to 'evaluation' for harder tasks
                    'task_id': task_id,
                    'reset_sol_grid': 'padding'
                }
            )
            
            episode_reward = 0
            done = False
            
            while not done:
                # Use greedy action selection for evaluation
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device)
                    action_logits, _ = self.agent.ac_network.forward(obs_tensor)
                    action = torch.argmax(action_logits, dim=1).item()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            eval_rewards.append(episode_reward)
            # Success if episode completed without early termination
            eval_success.append(1.0 if truncated and not terminated else 0.0)
        
        return {
            'eval_mean_reward': np.mean(eval_rewards),
            'eval_std_reward': np.std(eval_rewards),
            'eval_success_rate': np.mean(eval_success),
            'eval_episodes': num_episodes
        }
    
    def train(self):
        """Main training loop."""
        print("Starting PPO training...")
        print(f"Configuration: {self.config}")
        
        best_mean_reward = -float('inf')
        first = True
        for update in range(self.config.training.num_updates):
            start_time = time.time()
            
            # Collect rollouts
            rollout_info = self.collect_rollouts(self.config.training.rollout_steps, update)
            # Grid visualization
            total_reward = rollout_info['total_reward']
            if total_reward > -0.98 and update % self.config.logging.visualize_period == 0 and update > 0:
                print(f"Creating grid visualization at update {update}...")
                self.visualize_grid(update, first=first)
                first = False
            
            # Update agent
            training_metrics = self.agent.update(
                next_value=rollout_info['final_value'],
                gae_lambda=self.config.training.gae_lambda,
                ppo_epochs=self.config.training.ppo_epochs,
                mini_batch_size=self.config.training.mini_batch_size
            )
            
            update_time = time.time() - start_time
            
            # Logging
            if update % self.config.logging.log_interval == 0:
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                success_rate = np.mean(self.success_rate) if self.success_rate else 0
                
                print(f"\nUpdate {update}/{self.config.training.num_updates}")
                print(f"Episodes completed in rollout: {rollout_info['episodes_completed']}")
                print(f"Mean reward (last 100 episodes): {mean_reward:.3f}")
                print(f"Mean episode length: {mean_length:.1f}")
                print(f"Success rate: {success_rate:.3f}")
                print(f"Update time: {update_time:.2f}s")
                
                if training_metrics:
                    print(f"Policy loss: {training_metrics['policy_loss']:.4f}")
                    print(f"Value loss: {training_metrics['value_loss']:.4f}")
                    print(f"Entropy loss: {training_metrics['entropy_loss']:.4f}")
                
                # Log to wandb
                if self.config.logging.use_wandb:
                    log_dict = {
                        'train/mean_reward': mean_reward,
                        'train/mean_length': mean_length,
                        'train/success_rate': success_rate,
                        'train/episodes_completed': rollout_info['episodes_completed'],
                        'train/update_time': update_time,
                        'update': update
                    }
                    if training_metrics:
                        log_dict.update({
                            'train/policy_loss': training_metrics['policy_loss'],
                            'train/value_loss': training_metrics['value_loss'],
                            'train/entropy_loss': training_metrics['entropy_loss']
                        })
                    wandb.log(log_dict)
                
                # Log to tensorboard
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar('train/mean_reward', mean_reward, update)
                    self.tensorboard_writer.add_scalar('train/mean_length', mean_length, update)
                    self.tensorboard_writer.add_scalar('train/success_rate', success_rate, update)
                    self.tensorboard_writer.add_scalar('train/episodes_completed', rollout_info['episodes_completed'], update)
                    self.tensorboard_writer.add_scalar('train/update_time', update_time, update)
                    if training_metrics:
                        self.tensorboard_writer.add_scalar('train/policy_loss', training_metrics['policy_loss'], update)
                        self.tensorboard_writer.add_scalar('train/value_loss', training_metrics['value_loss'], update)
                        self.tensorboard_writer.add_scalar('train/entropy_loss', training_metrics['entropy_loss'], update)
            
            # Evaluation
            if update % self.config.logging.eval_interval == 0 and update > 0:
                eval_metrics = self.evaluate_agent(self.config.logging.eval_episodes)
                print(f"\nEvaluation after update {update}:")
                for key, value in eval_metrics.items():
                    print(f"{key}: {value:.3f}")
                
                # Log evaluation metrics
                if self.config.logging.use_wandb:
                    eval_log_dict = {f"eval/{key}": value for key, value in eval_metrics.items()}
                    eval_log_dict['update'] = update
                    wandb.log(eval_log_dict)
                
                if self.tensorboard_writer:
                    for key, value in eval_metrics.items():
                        self.tensorboard_writer.add_scalar(f"eval/{key.replace('eval_', '')}", value, update)
                
                # Save best model
                if eval_metrics['eval_mean_reward'] > best_mean_reward:
                    best_mean_reward = eval_metrics['eval_mean_reward']
                    self.agent.save(os.path.join(self.config.logging.save_dir, 'best_model.pth'))
                    print(f"New best model saved! Mean reward: {best_mean_reward:.3f}")
                    
                    if self.config.logging.use_wandb:
                        wandb.log({'train/best_mean_reward': best_mean_reward, 'update': update})
            

            # Save checkpoint
            if update % self.config.logging.save_interval == 0 and update > 0:
                checkpoint_path = os.path.join(self.config.logging.save_dir, f'checkpoint_{update}.pth')
                self.agent.save(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        print("Training completed!")
        
        # Final save
        final_path = os.path.join(self.config.logging.save_dir, 'final_model.pth')
        self.agent.save(final_path)
        print(f"Final model saved: {final_path}")
        
        # Close loggers
        if self.config.logging.use_wandb:
            wandb.finish()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()


@hydra.main(version_base=None, config_path="config", config_name="ppo")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    print("Training configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create save directory
    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    random.seed(cfg.environment.seed)
    np.random.seed(cfg.environment.seed)
    torch.manual_seed(cfg.environment.seed)
    
    # Create trainer and start training
    trainer = ArcAgiTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()