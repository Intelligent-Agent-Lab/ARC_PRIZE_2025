import torch.nn as nn 

class ActorCritic_MLP(nn.Module):
    """Actor-Critic network for PPO with grid sequence observations."""
    
    def __init__(self, input_size: int = 5400, hidden_size: int = 64, action_size: int = 11):
        super(ActorCritic_MLP, self).__init__()
        self.embedding = nn.Embedding(11, hidden_size)
        # ! [Batch, 5400, Hidden]
        # Shared layers
        self.shared_layers = nn.Sequential(
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
            nn.GELU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        """Forward pass returning both action probabilities and state value."""
        x = x.to(int)
        x = self.embedding(x)
        shared_features = self.shared_layers(x)
        shared_features = shared_features.mean(dim=1)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value