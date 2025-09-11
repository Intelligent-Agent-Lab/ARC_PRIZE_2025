import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List
import gymnasium as gym
from network.mlp import ActorCritic_MLP
from network.vit import ActorCritic_ViT

class RunningMeanStd:
    """Running mean and standard deviation for observation normalization."""
    
    def __init__(self, shape, epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

class PPOAgent:
    """PPO Agent for training on ArcAgiGrid environment."""
    
    def __init__(self, 
                 cfg,
                 input_size: int = 5400,
                 action_size: int = 9900,
                 hidden_size: int = 512,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 clip_range_vf: float = None,  # Value function clipping range
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 target_kl: float = 0.01,  # Target KL divergence for early stopping
                 normalize_obs: bool = True,
                 normalize_reward: bool = True,  # Reward normalization
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.clip_range_vf = clip_range_vf
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        
        # Initialize observation normalization
        if self.normalize_obs:
            self.obs_rms = RunningMeanStd(shape=(30, 180))
            
        # Initialize reward normalization
        if self.normalize_reward:
            self.reward_rms = RunningMeanStd(shape=())
        
        # Initialize actor-critic network
        if cfg.network.type == 'mlp':
            # Use embedding for discrete inputs, direct processing for normalized inputs
            use_embedding = not normalize_obs
            self.ac_network = ActorCritic_MLP(input_size, hidden_size, action_size, use_embedding).to(device)
        elif cfg.network.type == 'transformer':
            pass
        elif cfg.network.type == 'mamba':
            pass
        elif cfg.network.type == 'vit':
            self.ac_network = ActorCritic_ViT().to(device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)
        
        # Learning rate scheduler (cosine annealing)
        self.initial_lr = learning_rate
        self.scheduler = None  # Will be set by training script with total_steps
        
        # Storage for rollouts
        self.reset_storage()
        
    def setup_lr_scheduler(self, total_updates: int, schedule_type: str = 'linear'):
        """Setup learning rate scheduler."""
        if schedule_type == 'linear':
            # Linear decay to 0
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, 
                lambda step: max(0.0, 1.0 - step / total_updates)
            )
        elif schedule_type == 'cosine':
            # Cosine annealing
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=total_updates,
                eta_min=0.0
            )
        else:
            self.scheduler = None
    
    def update_lr_scheduler(self):
        """Update learning rate scheduler if available."""
        if self.scheduler is not None:
            self.scheduler.step()
            
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
        
    def get_action_and_value(self, x, action=None):
        """Get action, log probability, entropy, and value."""
        action_logits, value = self.ac_network(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value
        
    def reset_storage(self):
        """Reset storage for collecting rollout data."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running mean and std."""
        if self.normalize_obs:
            return self.obs_rms.normalize(obs)
        return obs
    
    def select_action(self, observation: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy."""
        
        # Normalize observation
        normalized_obs = self._normalize_observation(observation)
        # Convert to tensor
        obs_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, entropy, value = self.get_action_and_value(obs_tensor)
        return action, log_prob.item(), value.item()
    
    def store_transition(self, obs, action, log_prob, reward, value, done):
        """Store transition in rollout buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)  # Store raw reward, normalize during GAE computation
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float, gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (improved version)."""
        # Convert to numpy arrays for efficient computation
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        
        # Normalize rewards if enabled
        if self.normalize_reward and len(rewards) > 1:
            self.reward_rms.update(rewards)
            rewards = self.reward_rms.normalize(rewards)
        
        # Add next value for bootstrap
        next_values = np.append(values[1:], next_value)
        
        # Compute TD errors (deltas)
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Compute GAE advantages efficiently
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        
        for step in reversed(range(len(rewards))):
            gae = deltas[step] + self.gamma * gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
        
        # Compute returns (advantages + values)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, next_obs: np.ndarray = None, next_value: float = None, gae_lambda: float = 0.95, 
               ppo_epochs: int = 4, mini_batch_size: int = 64):
        """Update policy using PPO."""
        if len(self.observations) < 2:
            return {}
        
        # Compute next_value if not provided
        if next_value is None and next_obs is not None:
            # Normalize next observation if needed
            normalized_next_obs = self._normalize_observation(next_obs)
            next_obs_tensor = torch.FloatTensor(normalized_next_obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                _, _, _, next_value_tensor = self.get_action_and_value(next_obs_tensor)
                next_value = next_value_tensor.item()
        elif next_value is None:
            next_value = 0.0  # Fallback to 0 if no next_obs provided
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value, gae_lambda)
        
        # Update observation normalization statistics
        obs_array = np.array(self.observations)
        if self.normalize_obs:
            self.obs_rms.update(obs_array)
            normalized_obs = np.array([self.obs_rms.normalize(obs) for obs in obs_array])
        else:
            normalized_obs = obs_array
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(normalized_obs).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        old_values_tensor = torch.FloatTensor(self.values).to(self.device)  # Store old values for clipping
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_divergence = 0
        early_stop_epochs = 0
        
        # PPO update epochs
        for epoch in range(ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(obs_tensor))
            
            for start_idx in range(0, len(obs_tensor), mini_batch_size):
                end_idx = start_idx + mini_batch_size
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, values = self.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # Calculate ratios and KL divergence
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                kl_divergence = (batch_old_log_probs - new_log_probs).mean()  # Approximate KL
                
                # Policy loss (clipped)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with optional clipping
                values_pred = values.squeeze()
                
                if self.clip_range_vf is not None:
                    # Clipped value loss (like in OpenAI's PPO2)
                    values_pred_clipped = batch_old_values + torch.clamp(
                        values_pred - batch_old_values,
                        -self.clip_range_vf, self.clip_range_vf
                    )
                    value_loss_clipped = F.mse_loss(values_pred_clipped, batch_returns)
                    value_loss_unclipped = F.mse_loss(values_pred, batch_returns)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                else:
                    # Standard unclipped value loss
                    value_loss = F.mse_loss(values_pred, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), 0.5)
                self.optimizer.step()
                
                # Accumulate losses for logging
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_kl_divergence += kl_divergence.item()
                
            # Check for early stopping based on KL divergence
            avg_kl = total_kl_divergence / (len(obs_tensor) // mini_batch_size + 1)
            if self.target_kl is not None and avg_kl > self.target_kl:
                early_stop_epochs = epoch + 1
                print(f"Early stopping at epoch {epoch + 1}/{ppo_epochs} due to high KL divergence: {avg_kl:.4f} > {self.target_kl}")
                break
        
        # Reset storage
        self.reset_storage()
        
        # Return training metrics
        actual_epochs = early_stop_epochs if early_stop_epochs > 0 else ppo_epochs
        num_updates = actual_epochs * (len(obs_tensor) // mini_batch_size + 1)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'kl_divergence': total_kl_divergence / num_updates,
            'early_stopped': early_stop_epochs > 0,
            'epochs_completed': actual_epochs,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy_loss) / num_updates
        }
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        checkpoint_dict = {
            'model_state_dict': self.ac_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.normalize_obs:
            checkpoint_dict.update({
                'obs_mean': self.obs_rms.mean,
                'obs_var': self.obs_rms.var,
                'obs_count': self.obs_rms.count
            })
        if self.normalize_reward:
            checkpoint_dict.update({
                'reward_mean': self.reward_rms.mean,
                'reward_var': self.reward_rms.var,
                'reward_count': self.reward_rms.count
            })
        torch.save(checkpoint_dict, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.normalize_obs and 'obs_mean' in checkpoint:
            self.obs_rms.mean = checkpoint['obs_mean']
            self.obs_rms.var = checkpoint['obs_var']
            self.obs_rms.count = checkpoint['obs_count']
            
        if self.normalize_reward and 'reward_mean' in checkpoint:
            self.reward_rms.mean = checkpoint['reward_mean']
            self.reward_rms.var = checkpoint['reward_var']
            self.reward_rms.count = checkpoint['reward_count']