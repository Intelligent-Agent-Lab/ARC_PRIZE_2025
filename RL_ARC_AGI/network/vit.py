import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Patch embedding layer for ViT adapted to 30x180 input."""
    
    def __init__(self, grid_size=(30, 180), patch_size=15, embed_dim=768, vocab_size=11):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.n_patches = (grid_size[0] // patch_size) * (grid_size[1] // patch_size)
        
        # Token embedding for discrete values (0-10)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Patch projection
        self.patch_projection = nn.Linear(patch_size * patch_size * embed_dim, embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Position embedding
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape to grid: [batch, 30, 180]
        x = x.view(batch_size, self.grid_size[0], self.grid_size[1])
        
        # Convert to patches: [batch, n_patches, patch_size, patch_size]
        patches = []
        for i in range(0, self.grid_size[0], self.patch_size):
            for j in range(0, self.grid_size[1], self.patch_size):
                patch = x[:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        
        # Stack patches: [batch, n_patches, patch_size, patch_size]
        patches = torch.stack(patches, dim=1)
        
        # Token embedding: [batch, n_patches, patch_size, patch_size, embed_dim]
        patches = self.token_embedding(patches.long())
        
        # Flatten patches: [batch, n_patches, patch_size * patch_size * embed_dim]
        patches = patches.view(batch_size, self.n_patches, -1)
        
        # Project patches: [batch, n_patches, embed_dim]
        patches = self.patch_projection(patches)
        
        # Add class token: [batch, 1, embed_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate: [batch, n_patches + 1, embed_dim]
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # Add position embedding
        x = x + self.position_embedding
        
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class ActorCritic_ViT(nn.Module):
    """Vision Transformer based Actor-Critic network for PPO with grid observations."""
    
    def __init__(self, grid_size=(30, 180), patch_size=15, embed_dim=768, 
                 num_heads=12, num_layers=12, mlp_ratio=4, vocab_size=11, 
                 action_size=11, dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(grid_size, patch_size, embed_dim, vocab_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, action_size)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
    
    def forward(self, x):
        """Forward pass returning both action probabilities and state value."""
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Use CLS token (first token) for classification
        cls_token = x[:, 0]
        
        # Actor and critic heads
        action_logits = self.actor(cls_token)
        state_value = self.critic(cls_token)
        
        return action_logits, state_value