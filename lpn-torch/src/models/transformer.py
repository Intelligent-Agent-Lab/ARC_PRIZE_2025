from typing import Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

# Mock config classes - you'll need to replace these with your actual config classes
@dataclass
class TransformerLayerConfig:
    dropout_rate: float = 0.1
    use_bias: bool = True

@dataclass 
class EncoderTransformerConfig:
    vocab_size: int = 10
    max_rows: int = 30
    max_cols: int = 30
    emb_dim: int = 512
    latent_dim: int = 256
    num_layers: int = 6
    variational: bool = True
    scaled_position_embeddings: bool = False
    latent_projection_bias: bool = True
    dtype: torch.dtype = torch.float32
    transformer_layer: TransformerLayerConfig = TransformerLayerConfig()
    
    @property
    def max_len(self):
        return self.max_rows * self.max_cols

@dataclass
class DecoderTransformerConfig:
    vocab_size: int = 10
    output_vocab_size: int = 10
    max_rows: int = 30
    max_cols: int = 30
    emb_dim: int = 512
    num_layers: int = 6
    scaled_position_embeddings: bool = False
    next_position_embeddings: bool = False
    next_position_embeddings_new_input_embeds: bool = False
    logits_projection_bias: bool = True
    dtype: torch.dtype = torch.float32
    transformer_layer: TransformerLayerConfig = TransformerLayerConfig()
    
    @property
    def max_len(self):
        return self.max_rows * self.max_cols

# Mock TransformerLayer - you'll need to replace this with your actual implementation
class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        self.config = config
        # This is a placeholder - implement your actual transformer layer
        
    def forward(self, embeddings: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, 
                training: bool = True) -> torch.Tensor:
        # Placeholder implementation
        return embeddings


class EncoderTransformer(nn.Module):
    def __init__(self, config: EncoderTransformerConfig):
        super().__init__()
        self.config = config
        
        # Position embedding layers
        if config.scaled_position_embeddings:
            self.pos_row_embed = nn.Embedding(1, config.emb_dim)
            self.pos_col_embed = nn.Embedding(1, config.emb_dim)
        else:
            self.pos_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
            self.pos_col_embed = nn.Embedding(config.max_cols, config.emb_dim)
        
        # Color embedding
        self.colors_embed = nn.Embedding(config.vocab_size, config.emb_dim)
        
        # Channel embedding
        self.channels_embed = nn.Embedding(2, config.emb_dim)
        
        # Grid shapes embedding
        self.grid_shapes_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
        self.grid_shapes_col_embed = nn.Embedding(config.max_cols, config.emb_dim)
        
        # CLS token
        self.cls_token = nn.Embedding(1, config.emb_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config.transformer_layer) for _ in range(config.num_layers)
        ])
        
        # Dropout
        self.embed_dropout = nn.Dropout(config.transformer_layer.dropout_rate)
        
        # Output layers
        self.cls_layer_norm = nn.LayerNorm(config.emb_dim, bias=config.transformer_layer.use_bias)
        self.latent_mu_proj = nn.Linear(config.emb_dim, config.latent_dim, bias=config.latent_projection_bias)
        
        if config.variational:
            self.latent_logvar_proj = nn.Linear(config.emb_dim, config.latent_dim, bias=config.latent_projection_bias)

    def forward(
        self,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Applies Transformer Encoder on the (input, output) pairs.

        Args:
            pairs: input data as tokens. Shape (*B, R, C, 2).
                - R: number of rows.
                - C: number of columns.
                - 2: two channels (input and output)
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, 2, 2). The last two dimension
                represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].
                Expects grid shapes values to be in [1, max_rows] and [1, max_cols].
            training: if True dropout is applied otherwise it is not.

        Returns:
            latent_mu: output of shape (*B, H) representing the mean latent embeddings of the (input, output)
                pairs.
            latent_logvar: output of shape (*B, H) representing the log-variance of the latent embeddings of
                the (input, output) pairs.
        """
        
        x = self.embed_grids(pairs, grid_shapes, training)

        # Transformer block.
        pad_mask = self.make_pad_mask(grid_shapes)
        for layer in self.transformer_layers:
            x = layer(embeddings=x, pad_mask=pad_mask, training=training)

        # Extract the CLS embedding.
        cls_embed = x[..., 0, :]
        # Project the cls embedding to the program space.
        cls_embed = self.cls_layer_norm(cls_embed)
        
        latent_mu = self.latent_mu_proj(cls_embed).float()
        
        if self.config.variational:
            latent_logvar = self.latent_logvar_proj(cls_embed).float()
        else:
            latent_logvar = None

        return latent_mu, latent_logvar

    def embed_grids(self, pairs: torch.Tensor, grid_shapes: torch.Tensor, training: bool) -> torch.Tensor:
        config = self.config
        device = pairs.device

        # Position embedding block.
        if config.scaled_position_embeddings:
            pos_row_embed = self.pos_row_embed(torch.zeros(config.max_rows, dtype=torch.long, device=device))
            pos_col_embed = self.pos_col_embed(torch.zeros(config.max_cols, dtype=torch.long, device=device))
            row_indices = torch.arange(1, config.max_rows + 1, device=device).unsqueeze(1)
            col_indices = torch.arange(1, config.max_cols + 1, device=device).unsqueeze(1)
            pos_row_embeds = row_indices * pos_row_embed
            pos_col_embeds = col_indices * pos_col_embed
            pos_embed = pos_row_embeds.unsqueeze(1).unsqueeze(2) + pos_col_embeds.unsqueeze(0).unsqueeze(2)
        else:
            pos_row_embed = self.pos_row_embed(torch.arange(config.max_rows, dtype=torch.long, device=device))
            pos_col_embed = self.pos_col_embed(torch.arange(config.max_cols, dtype=torch.long, device=device))
            pos_embed = pos_row_embed.unsqueeze(1).unsqueeze(2) + pos_col_embed.unsqueeze(0).unsqueeze(2)

        # Colors embedding block.
        colors_embed = self.colors_embed(pairs)

        # Channels embedding block.
        channels_embed = self.channels_embed(torch.arange(2, dtype=torch.long, device=device))

        # Combine all the embeddings into a sequence x of shape (*B, 1+2*(R*C), H)
        x = colors_embed + pos_embed + channels_embed
        # Flatten the rows, columns and channels.
        x = x.view(*x.shape[:-4], -1, x.shape[-1])  # (*B, 2*R*C, H)

        # Embed the grid shape tokens.
        grid_shapes_row_embed = self.grid_shapes_row_embed(grid_shapes[..., 0, :] - 1)
        grid_shapes_row_embed = grid_shapes_row_embed + channels_embed
        grid_shapes_col_embed = self.grid_shapes_col_embed(grid_shapes[..., 1, :] - 1)
        grid_shapes_col_embed = grid_shapes_col_embed + channels_embed
        grid_shapes_embed = torch.cat([grid_shapes_row_embed, grid_shapes_col_embed], dim=-2)
        x = torch.cat([grid_shapes_embed, x], dim=-2)  # (*B, 4+2*R*C, H)

        # Add the cls token.
        batch_shape = x.shape[:-2]
        cls_token = self.cls_token(torch.zeros(*batch_shape, 1, dtype=torch.long, device=device))
        x = torch.cat([cls_token, x], dim=-2)  # (*B, 1+4+2*R*C, H)
        
        assert x.shape[-2] == 1 + 4 + 2 * config.max_len  # 1805
        
        if training:
            x = self.embed_dropout(x)
        return x

    def make_pad_mask(self, grid_shapes: torch.Tensor) -> torch.Tensor:
        """Make the pad mask False outside of the grid shapes and True inside.

        Args:
            grid_shapes: shapes of the grids (e.g. 30x30). Shape (*B, 2, 2). The last two dimension
                represents (rows, columns) of two channels, e.g. [[R_input, R_output], [C_input, C_output]].

        Returns:
            pad mask of shape (*B, 1, T, T) with T = 1 + 4 + 2 * max_rows * max_cols.
        """
        device = grid_shapes.device
        batch_dims = grid_shapes.shape[:-2]
        
        row_arange = torch.arange(self.config.max_rows, device=device).view(
            *([1] * len(batch_dims)), self.config.max_rows, 1
        )
        row_mask = row_arange < grid_shapes[..., 0:1, :]
        
        col_arange = torch.arange(self.config.max_cols, device=device).view(
            *([1] * len(batch_dims)), self.config.max_cols, 1
        )
        col_mask = col_arange < grid_shapes[..., 1:2, :]
        
        pad_mask = row_mask.unsqueeze(-2) & col_mask.unsqueeze(-3)
        # Flatten the rows, columns and channels.
        pad_mask = pad_mask.view(*pad_mask.shape[:-3], 1, -1)
        # Add the masks corresponding to the cls token and grid shapes tokens.
        ones_mask = torch.ones(*pad_mask.shape[:-1], 1 + 4, dtype=torch.bool, device=device)
        pad_mask = torch.cat([ones_mask, pad_mask], dim=-1)
        # Outer product to make the self-attention mask.
        pad_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(-2)
        return pad_mask


class DecoderTransformer(nn.Module):
    def __init__(self, config: DecoderTransformerConfig):
        super().__init__()
        self.config = config
        
        # Context embedding
        self.context_embed = nn.Linear(config.emb_dim, config.emb_dim, bias=config.transformer_layer.use_bias)
        
        # Position embedding layers
        if config.scaled_position_embeddings:
            self.pos_row_embed = nn.Embedding(1, config.emb_dim)
            self.pos_col_embed = nn.Embedding(1, config.emb_dim)
            if config.next_position_embeddings and config.next_position_embeddings_new_input_embeds:
                self.input_pos_row_embed = nn.Embedding(1, config.emb_dim)
                self.input_pos_col_embed = nn.Embedding(1, config.emb_dim)
        else:
            self.pos_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
            self.pos_col_embed = nn.Embedding(config.max_cols, config.emb_dim)
            if config.next_position_embeddings and config.next_position_embeddings_new_input_embeds:
                self.input_pos_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
                self.input_pos_col_embed = nn.Embedding(config.max_cols, config.emb_dim)
        
        # Grid shapes embedding
        self.grid_shapes_row_embed = nn.Embedding(config.max_rows, config.emb_dim)
        self.grid_shapes_col_embed = nn.Embedding(config.max_cols, config.emb_dim)
        
        # Colors embedding
        self.colors_embed = nn.Embedding(config.vocab_size, config.emb_dim)
        
        # Input/Output embedding
        self.input_output_embed = nn.Embedding(2, config.emb_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config.transformer_layer) for _ in range(config.num_layers)
        ])
        
        # Dropout
        self.embed_dropout = nn.Dropout(config.transformer_layer.dropout_rate)
        
        # Output layers
        self.row_logits_layer_norm = nn.LayerNorm(config.emb_dim, bias=config.transformer_layer.use_bias)
        self.col_logits_layer_norm = nn.LayerNorm(config.emb_dim, bias=config.transformer_layer.use_bias)
        self.grid_logits_layer_norm = nn.LayerNorm(config.emb_dim, bias=config.transformer_layer.use_bias)
        
        self.shape_row_logits_proj = nn.Linear(config.emb_dim, config.max_rows, bias=config.logits_projection_bias)
        self.shape_col_logits_proj = nn.Linear(config.emb_dim, config.max_cols, bias=config.logits_projection_bias)
        self.grid_logits_proj = nn.Linear(config.emb_dim, config.output_vocab_size, bias=config.logits_projection_bias)

    def forward(
        self,
        input_seq: torch.Tensor,
        output_seq: torch.Tensor,
        context: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies Transformer Decoder on the task outputs to reconstruct them given a context latent.

        Args:
            input_seq: flattened task input grid as tokens. Shape (*B, 2+R*C).
            output_seq: flattened task output grid as tokens. Shape (*B, 2+R*C).
            context: latent program of the task. Shape (*B, H).
            training: if True dropout is applied otherwise it is not.

        Returns:
            grid_shape_row_logits of shape (*B, R) representing the logits for the grid shape row.
            grid_shape_col_logits of shape (*B, C) representing the logits for the grid shape column.
            output_grid_logits of shape (*B, R*C, V) representing the logits of the next-token predictions.
        """
        x = self.embed_inputs(input_seq, output_seq, context, training)

        # Transformer block.
        causal_pad_mask = self.make_causal_pad_mask(
            input_grid_shape=input_seq[..., :2], output_grid_shape=output_seq[..., :2]
        )
        for layer in self.transformer_layers:
            x = layer(embeddings=x, pad_mask=causal_pad_mask, training=training)

        grid_shape_row_logits, grid_shape_col_logits, output_grid_logits = self.extract_logits(
            x, input_seq.shape[-1]
        )
        # Cast the output back to float32.
        grid_shape_row_logits = grid_shape_row_logits.float()
        grid_shape_col_logits = grid_shape_col_logits.float()
        output_grid_logits = output_grid_logits.float()

        return grid_shape_row_logits, grid_shape_col_logits, output_grid_logits

    def embed_inputs(
        self, input_seq: torch.Tensor, output_seq: torch.Tensor, context: torch.Tensor, training: bool
    ) -> torch.Tensor:
        config = self.config
        device = input_seq.device

        # Context embedding block.
        context_embed = self.context_embed(context)

        # Position embedding block.
        if config.scaled_position_embeddings:
            pos_row_embed = self.pos_row_embed(torch.zeros(config.max_rows, dtype=torch.long, device=device))
            pos_col_embed = self.pos_col_embed(torch.zeros(config.max_cols, dtype=torch.long, device=device))
            row_indices = torch.arange(1, config.max_rows + 1, device=device).unsqueeze(1)
            col_indices = torch.arange(1, config.max_cols + 1, device=device).unsqueeze(1)
            pos_row_embeds = row_indices * pos_row_embed
            pos_col_embeds = col_indices * pos_col_embed
            pos_embed = pos_row_embeds.unsqueeze(1) + pos_col_embeds.unsqueeze(0)
        else:
            pos_row_embed = self.pos_row_embed(torch.arange(config.max_rows, dtype=torch.long, device=device))
            pos_col_embed = self.pos_col_embed(torch.arange(config.max_cols, dtype=torch.long, device=device))
            pos_embed = pos_row_embed.unsqueeze(1) + pos_col_embed.unsqueeze(0)

        if config.next_position_embeddings:
            input_num_cols, output_num_cols = input_seq[..., 1], output_seq[..., 1]
            shifted_left_pos_embed = torch.roll(pos_embed, shifts=-1, dims=-2)
            first_col_embed = pos_embed[:, 0, :]
            shifted_up_first_col_embed = torch.roll(first_col_embed, shifts=-1, dims=-2)
            batch_dims = len(input_num_cols.shape)
            arange_broadcast = torch.arange(config.max_cols, device=device).view(
                *([1] * batch_dims), config.max_cols
            )
            
            if config.next_position_embeddings_new_input_embeds:
                # Generate new position embeddings for the input tokens only.
                if config.scaled_position_embeddings:
                    input_pos_row_embed = self.input_pos_row_embed(torch.zeros(config.max_rows, dtype=torch.long, device=device))
                    input_pos_col_embed = self.input_pos_col_embed(torch.zeros(config.max_cols, dtype=torch.long, device=device))
                    input_pos_row_embeds = row_indices * input_pos_row_embed
                    input_pos_col_embeds = col_indices * input_pos_col_embed
                    input_pos_embeds = input_pos_row_embeds.unsqueeze(1) + input_pos_col_embeds.unsqueeze(0)
                else:
                    input_pos_row_embed = self.input_pos_row_embed(torch.arange(config.max_rows, dtype=torch.long, device=device))
                    input_pos_col_embed = self.input_pos_col_embed(torch.arange(config.max_cols, dtype=torch.long, device=device))
                    input_pos_embeds = input_pos_row_embed.unsqueeze(1) + input_pos_col_embed.unsqueeze(0)
            else:
                # Reuse the position embeddings for the input tokens.
                input_pos_embeds = pos_embed

            condition = (arange_broadcast == output_num_cols.unsqueeze(-1) - 1).unsqueeze(-2).unsqueeze(-1)
            output_pos_embeds = torch.where(
                condition,
                shifted_up_first_col_embed.unsqueeze(1),
                shifted_left_pos_embed,
            )
            input_pos_embeds = input_pos_embeds.view(*input_pos_embeds.shape[:-3], -1, config.emb_dim)
            output_pos_embeds = output_pos_embeds.view(*output_pos_embeds.shape[:-3], -1, config.emb_dim)
        else:
            pos_embeds = pos_embed.view(-1, config.emb_dim)
            input_pos_embeds, output_pos_embeds = pos_embeds, pos_embeds

        # Grid shapes embedding block.
        input_grid_shapes_row_embed = self.grid_shapes_row_embed(input_seq[..., 0] - 1)
        output_grid_shapes_row_embed = self.grid_shapes_row_embed(output_seq[..., 0] - 1)
        input_grid_shapes_col_embed = self.grid_shapes_col_embed(input_seq[..., 1] - 1)
        output_grid_shapes_col_embed = self.grid_shapes_col_embed(output_seq[..., 1] - 1)

        # Colors embedding block.
        input_colors_embed = self.colors_embed(input_seq[..., 2:])
        output_colors_embed = self.colors_embed(output_seq[..., 2:])
        input_embed, output_embed = self.input_output_embed(torch.arange(2, dtype=torch.long, device=device))

        # Combining all the embeddings into a sequence x of shape (*B, 1+2*(2+R*C), H)
        x_input_shape_row = (input_grid_shapes_row_embed + input_embed).unsqueeze(-2)
        x_input_shape_col = (input_grid_shapes_col_embed + input_embed).unsqueeze(-2)
        x_input_colors = input_colors_embed + input_pos_embeds + input_embed
        x_context = context_embed.unsqueeze(-2)
        x_output_shape_row = (output_grid_shapes_row_embed + output_embed).unsqueeze(-2)
        x_output_shape_col = (output_grid_shapes_col_embed + output_embed).unsqueeze(-2)
        x_output_colors = output_colors_embed + output_pos_embeds + output_embed
        
        x = torch.cat([
            x_input_shape_row,
            x_input_shape_col,
            x_input_colors,
            x_context,
            x_output_shape_row,
            x_output_shape_col,
            x_output_colors,
        ], dim=-2)
        
        if training:
            x = self.embed_dropout(x)
        assert x.shape[-2] == 1 + 2 * (2 + config.max_len)  # 1805
        return x

    def make_causal_pad_mask(self, input_grid_shape: torch.Tensor, output_grid_shape: torch.Tensor) -> torch.Tensor:
        """Make a mask for causal attention with proper padding."""
        device = input_grid_shape.device
        batch_dims = input_grid_shape.shape[:-1]
        
        row_arange = torch.arange(self.config.max_rows, device=device).view(
            *([1] * len(batch_dims)), self.config.max_rows
        )
        col_arange = torch.arange(self.config.max_cols, device=device).view(
            *([1] * len(batch_dims)), self.config.max_cols
        )

        # Input pad mask
        input_row_mask = row_arange < input_grid_shape[..., :1]
        input_col_mask = col_arange < input_grid_shape[..., 1:]
        input_pad_mask = input_row_mask.unsqueeze(-1) & input_col_mask.unsqueeze(-2)
        # Flatten the rows and columns.
        input_pad_mask = input_pad_mask.view(*input_pad_mask.shape[:-2], 1, -1)
        # Add the masks corresponding to the input grid shapes tokens.
        input_ones = torch.ones(*input_pad_mask.shape[:-1], 2, dtype=torch.bool, device=device)
        input_pad_mask = torch.cat([input_ones, input_pad_mask], dim=-1)
        # Outer product to make the self-attention mask.
        input_input_pad_mask = input_pad_mask.unsqueeze(-1) & input_pad_mask.unsqueeze(-2)

        # Output pad mask
        output_row_mask = row_arange < output_grid_shape[..., :1]
        output_col_mask = col_arange < output_grid_shape[..., 1:]
        output_pad_mask = output_row_mask.unsqueeze(-1) & output_col_mask.unsqueeze(-2)
        # Flatten the rows and columns.
        output_pad_mask = output_pad_mask.view(*output_pad_mask.shape[:-2], 1, -1)
        # Add the masks corresponding to the output grid shapes tokens and the context.
        output_ones = torch.ones(*output_pad_mask.shape[:-1], 1 + 2, dtype=torch.bool, device=device)
        output_pad_mask = torch.cat([output_ones, output_pad_mask], dim=-1)
        # Outer product to make the self-attention mask.
        output_output_pad_mask = output_pad_mask.unsqueeze(-1) & output_pad_mask.unsqueeze(-2)

        # Output causal mask
        seq_len = output_output_pad_mask.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        output_output_causal_mask = causal_mask.view(*([1] * len(batch_dims)), 1, seq_len, seq_len)

        # Putting all masks together
        input_input_mask = input_input_pad_mask
        output_output_mask = output_output_pad_mask & output_output_causal_mask
        input_output_mask = torch.zeros_like(output_output_mask)[..., :-1, :]
        # make the input see the first token of the output (i.e. the context)
        input_output_mask = input_output_mask.clone()
        input_output_mask[..., 0, :] = input_pad_mask
        output_input_mask = output_pad_mask.unsqueeze(-1) & input_pad_mask.unsqueeze(-2)
        
        causal_pad_mask = torch.cat([
            torch.cat([input_input_mask, input_output_mask], dim=-1),
            torch.cat([output_input_mask, output_output_mask], dim=-1),
        ], dim=-2)
        return causal_pad_mask

    def extract_logits(
        self, x: torch.Tensor, input_seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.config

        # Keep the second half of the sequence (the output part) and remove the last token. Apply layer norm.
        shape_row_embeds = self.row_logits_layer_norm(x[..., input_seq_length, :])
        shape_col_embeds = self.col_logits_layer_norm(x[..., input_seq_length + 1, :])
        grid_embeds = self.grid_logits_layer_norm(x[..., input_seq_length + 2:-1, :])

        # Last projection to the different logits vocab sizes.
        shape_row_logits = self.shape_row_logits_proj(shape_row_embeds)
        shape_col_logits = self.shape_col_logits_proj(shape_col_embeds)
        grid_logits = self.grid_logits_proj(grid_embeds)
        return shape_row_logits, shape_col_logits, grid_logits


if __name__ == "__main__":
    import torch

    batch_size = 4
    mini_batch_size = 3
    max_rows = 30
    max_cols = 30
    vocab_size = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformer Encoder.
    encoder_config = EncoderTransformerConfig(
        vocab_size=vocab_size, max_rows=max_rows, max_cols=max_cols, variational=True
    )
    encoder = EncoderTransformer(encoder_config).to(device)

    pairs = torch.randint(
        0, vocab_size,
        (batch_size, mini_batch_size, max_rows, max_cols, 2),
        device=device
    )
    grid_shapes = torch.full((batch_size, mini_batch_size, 2, 2), 15, dtype=torch.long, device=device)
    
    num_parameters = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder -> number of parameters: {num_parameters:,}")
    
    print("Input shape:", pairs.shape, grid_shapes.shape)
    with torch.no_grad():
        latent_mu, latent_logvar = encoder(pairs, grid_shapes, training=False)
    
    assert latent_mu.shape == (batch_size, mini_batch_size, encoder_config.latent_dim)
    if latent_logvar is not None:
        print("Output shape (latent_mu):", latent_mu.shape)
        print("Output shape (latent_logvar):", latent_logvar.shape)
        assert latent_logvar.shape == (batch_size, mini_batch_size, encoder_config.latent_dim)
    else:
        print("Output shape:", latent_mu.shape)

    # Transformer Decoder.
    decoder_config = DecoderTransformerConfig(
        vocab_size=vocab_size, output_vocab_size=vocab_size, max_rows=max_rows, max_cols=max_cols
    )
    decoder = DecoderTransformer(decoder_config).to(device)

    inputs = torch.randint(
        0, vocab_size,
        (batch_size, max_rows, max_cols),
        device=device
    )
    inputs_grid_shapes = torch.full((batch_size, 2), 15, dtype=torch.long, device=device)
    flattened_input = inputs.view(*inputs.shape[:-2], -1)
    input_seq = torch.cat([inputs_grid_shapes, flattened_input], dim=-1)
    output_seq = torch.zeros_like(input_seq)
    output_seq[..., :2] = 1  # Initialize the grid shape tokens to 1.
    context = torch.randn(batch_size, encoder_config.latent_dim, device=device)
    
    num_parameters = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder -> number of parameters: {num_parameters:,}")
    
    print("Input shape:", inputs.shape)
    with torch.no_grad():
        row_logits, col_logits, logits = decoder(
            input_seq, output_seq, context, training=False
        )
    print("Output shape:", row_logits.shape, col_logits.shape, logits.shape)
    assert row_logits.shape == (batch_size, max_rows)
    assert col_logits.shape == (batch_size, max_cols)
    assert logits.shape == (batch_size, max_rows * max_cols, vocab_size)