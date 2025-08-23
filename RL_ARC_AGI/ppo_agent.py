import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List
import gymnasium as gym


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with grid sequence observations."""
    
    def __init__(self, input_size: int = 5400, hidden_size: int = 512, action_size: int = 11):
        super(ActorCritic, self).__init__()
        self.embedding = nn.Embedding(11, hidden_size)
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Embedding(11, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        """Forward pass returning both action probabilities and state value."""
        x = x.to(int)
        shared_features = self.shared_layers(x)
        shared_features = shared_features.mean(dim=1)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value
    
    def get_action_and_value(self, x, action=None):
        """Get action, log probability, entropy, and value."""
        action_logits, value = self.forward(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


class PPOAgent:
    """PPO Agent for training on ArcAgiGrid environment."""
    
    def __init__(self, 
                 input_size: int = 5400,
                 action_size: int = 11,
                 hidden_size: int = 512,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Initialize actor-critic network
        self.ac_network = ActorCritic(input_size, hidden_size, action_size).to(device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)
        
        # Storage for rollouts
        self.reset_storage()
        
    def reset_storage(self):
        """Reset storage for collecting rollout data."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def select_action(self, observation: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.ac_network.get_action_and_value(obs_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, obs, action, log_prob, reward, value, done):
        """Store transition in rollout buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float, gae_lambda: float = 0.95) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        # Add next value for bootstrap
        values = self.values + [next_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
        
        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self, next_value: float = 0.0, gae_lambda: float = 0.95, 
               ppo_epochs: int = 4, mini_batch_size: int = 64):
        """Update policy using PPO."""
        if len(self.observations) == 0:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value, gae_lambda)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
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
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, values = self.ac_network.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # Calculate ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Policy loss (clipped)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
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
        
        # Reset storage
        self.reset_storage()
        
        # Return training metrics
        num_updates = ppo_epochs * (len(obs_tensor) // mini_batch_size + 1)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy_loss) / num_updates
        }
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.ac_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])