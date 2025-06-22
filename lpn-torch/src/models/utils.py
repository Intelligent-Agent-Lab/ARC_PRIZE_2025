"""Adapted from https://github.com/google/flax/blob/main/examples/nlp_seq/models.py"""

from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerLayerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    num_heads: int = 8
    emb_dim_per_head: int = 16
    mlp_dim_factor: float = 4.0
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    use_bias: bool = False
    activation: str = "silu"
    dtype: torch.dtype = torch.float32
    emb_dim: int = field(default=None)

    def __post_init__(self):
        if self.emb_dim is None:
            object.__setattr__(self, "emb_dim", self.num_heads * self.emb_dim_per_head)


@dataclass
class EncoderTransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    transformer_layer: TransformerLayerConfig = field(default_factory=TransformerLayerConfig)
    vocab_size: int = 10
    output_vocab_size: int = 10
    num_layers: int = 2
    latent_dim: int = 32
    variational: bool = False
    max_rows: int = 30
    max_cols: int = 30
    latent_projection_bias: bool = False
    scaled_position_embeddings: bool = False
    dtype: torch.dtype = field(default=None)
    emb_dim: int = field(default=None)
    max_len: int = field(default=None)

    def __post_init__(self):
        if self.dtype is None:
            object.__setattr__(self, "dtype", self.transformer_layer.dtype)
        if self.emb_dim is None:
            object.__setattr__(self, "emb_dim", self.transformer_layer.emb_dim)
        if self.max_len is None:
            object.__setattr__(self, "max_len", self.max_rows * self.max_cols)


@dataclass
class DecoderTransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    transformer_layer: TransformerLayerConfig = field(default_factory=TransformerLayerConfig)
    vocab_size: int = 10
    output_vocab_size: int = 10
    num_layers: int = 2
    max_rows: int = 30
    max_cols: int = 30
    scaled_position_embeddings: bool = False
    next_position_embeddings: bool = True
    next_position_embeddings_new_input_embeds: bool = False
    logits_projection_bias: bool = False
    dtype: torch.dtype = field(default=None)
    emb_dim: int = field(default=None)
    max_len: int = field(default=None)

    def __post_init__(self):
        if self.dtype is None:
            object.__setattr__(self, "dtype", self.transformer_layer.dtype)
        if self.emb_dim is None:
            object.__setattr__(self, "emb_dim", self.transformer_layer.emb_dim)
        if self.max_len is None:
            object.__setattr__(self, "max_len", self.max_rows * self.max_cols)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
        config: TransformerLayerConfig dataclass containing hyperparameters.
    """

    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        self.config = config
        
        # Setup activation function
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")
        
        # Setup layers
        self.dense1 = nn.Linear(
            config.emb_dim, 
            int(config.mlp_dim_factor * config.emb_dim), 
            bias=config.use_bias
        )
        self.dense2 = nn.Linear(
            int(config.mlp_dim_factor * config.emb_dim), 
            config.emb_dim, 
            bias=config.use_bias
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, inputs: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Applies Transformer MlpBlock module."""
        x = inputs
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        if training:
            x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.emb_dim = config.emb_dim
        self.head_dim = config.emb_dim_per_head
        
        assert self.emb_dim == self.num_heads * self.head_dim
        
        self.qkv_proj = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=config.use_bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout_rate)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * emb_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 4:  # (batch_size, 1, seq_len, seq_len)
                mask = mask.expand(-1, self.num_heads, -1, -1)
            elif mask.dim() == 3:  # (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # Convert boolean mask to additive mask
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                scores = scores + mask
        
        # Apply softmax
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        
        # Apply attention dropout
        if training:
            attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        
        # Final projection
        out = self.out_proj(out)
        
        return out


class TransformerLayer(nn.Module):
    """Transformer encoder layer.

    Attributes:
        config: TransformerLayerConfig dataclass containing hyperparameters.
    """

    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        self.config = config
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.emb_dim, bias=config.use_bias)
        self.norm2 = nn.LayerNorm(config.emb_dim, bias=config.use_bias)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(config)
        
        # MLP block
        self.mlp = MlpBlock(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        embeddings: torch.Tensor,
        training: bool = True,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies TransformerLayer module.

        Args:
            embeddings: input embeddings.
            training: if True dropout is applied otherwise it is not.
            pad_mask: mask to apply on the inputs to avoid attending to padding tokens.

        Returns:
            output after transformer encoder layer.
        """
        # Attention block with pre-norm
        x = self.norm1(embeddings)
        x = self.attention(x, mask=pad_mask, training=training)
        if training:
            x = self.dropout(x)
        embeddings = embeddings + x  # Residual connection

        # MLP block with pre-norm
        x = self.norm2(embeddings)
        x = self.mlp(x, training=training)
        embeddings = embeddings + x  # Residual connection
        
        return embeddings


# Test the implementation
if __name__ == "__main__":
    # Test configuration
    config = TransformerLayerConfig(
        num_heads=8,
        emb_dim_per_head=64,
        mlp_dim_factor=4.0,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        use_bias=True,
        activation="silu"
    )
    
    print(f"Config embedding dimension: {config.emb_dim}")
    
    # Test MLP block
    mlp = MlpBlock(config)
    x = torch.randn(2, 10, config.emb_dim)
    output = mlp(x, training=True)
    print(f"MLP input shape: {x.shape}")
    print(f"MLP output shape: {output.shape}")
    
    # Test MultiHeadAttention
    attention = MultiHeadAttention(config)
    output = attention(x, training=True)
    print(f"Attention input shape: {x.shape}")
    print(f"Attention output shape: {output.shape}")
    
    # Test TransformerLayer
    transformer = TransformerLayer(config)
    output = transformer(x, training=True)
    print(f"Transformer input shape: {x.shape}")
    print(f"Transformer output shape: {output.shape}")
    
    # Test with mask
    mask = torch.ones(2, 1, 10, 10, dtype=torch.bool)
    mask[:, :, :, 5:] = False  # Mask out last 5 positions
    output_masked = transformer(x, training=True, pad_mask=mask)
    print(f"Transformer with mask output shape: {output_masked.shape}")
    
    # Test encoder config
    encoder_config = EncoderTransformerConfig(
        transformer_layer=config,
        vocab_size=1000,
        num_layers=6,
        latent_dim=256,
        variational=True
    )
    print(f"Encoder config max_len: {encoder_config.max_len}")
    print(f"Encoder config emb_dim: {encoder_config.emb_dim}")
    
    # Test decoder config
    decoder_config = DecoderTransformerConfig(
        transformer_layer=config,
        vocab_size=1000,
        num_layers=6
    )
    print(f"Decoder config max_len: {decoder_config.max_len}")
    print(f"Decoder config emb_dim: {decoder_config.emb_dim}")