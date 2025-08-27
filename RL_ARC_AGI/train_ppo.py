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

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from arc_agi_grid_env import ArcAgiGridEnv
from env_wrappers import create_wrapped_env
from ppo_agent import PPOAgent


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
        base_env = ArcAgiGridEnv(
            training_challenges_json=self.config.environment.training_challenges_json,
            training_solutions_json=self.config.environment.training_solutions_json,
            evaluation_challenges_json=self.config.environment.evaluation_challenges_json,
            evaluation_solutions_json=self.config.environment.evaluation_solutions_json,
            test_challenges_json=self.config.environment.test_challenges_json
        )
        
        # Wrap environment
        self.env = create_wrapped_env(
            base_env, 
            normalize=self.config.environment.normalize_obs,
            reward_shaping=self.config.environment.reward_shaping
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
        
    def collect_rollouts(self, rollout_steps: int) -> Dict[str, float]:
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
            done = terminated or truncated
            
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
        
        for update in range(self.config.training.num_updates):
            start_time = time.time()
            
            # Collect rollouts
            rollout_info = self.collect_rollouts(self.config.training.rollout_steps)
            
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
            
            # Evaluation
            if update % self.config.logging.eval_interval == 0 and update > 0:
                eval_metrics = self.evaluate_agent(self.config.logging.eval_episodes)
                print(f"\nEvaluation after update {update}:")
                for key, value in eval_metrics.items():
                    print(f"{key}: {value:.3f}")
                
                # Save best model
                if eval_metrics['eval_mean_reward'] > best_mean_reward:
                    best_mean_reward = eval_metrics['eval_mean_reward']
                    self.agent.save(os.path.join(self.config.logging.save_dir, 'best_model.pth'))
                    print(f"New best model saved! Mean reward: {best_mean_reward:.3f}")
            
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


@hydra.main(version_base=None, config_path="config", config_name="ppo")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    print("Training configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create save directory
    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create trainer and start training
    trainer = ArcAgiTrainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()