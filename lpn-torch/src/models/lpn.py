from typing import Literal, Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.transformer import EncoderTransformer, DecoderTransformer
from src.models.utils import EncoderTransformerConfig, DecoderTransformerConfig
from src.data_utils import make_leave_one_out


class LPN(nn.Module):
    def __init__(self, encoder: EncoderTransformer, decoder: DecoderTransformer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        training: bool = True,
        mode: Literal["mean", "all", "random_search", "gradient_ascent"] = "mean",
        prior_kl_coeff: Optional[float] = None,
        pairwise_kl_coeff: Optional[float] = None,
        **mode_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the LPN model.

        Args:
            pairs: input data as tokens. Shape (*B, N, R, C, 2).
            grid_shapes: shapes of the grids. Shape (*B, N, 2, 2).
            training: if True dropout is applied otherwise it is not.
            mode: mode of the forward pass.
            prior_kl_coeff: KL divergence coefficient for the variational inference.
            pairwise_kl_coeff: KL divergence coefficient for the pairwise KL divergence.
            mode_kwargs: additional keyword arguments for the inference mode.

        Returns:
            loss: loss value.
            metrics: dictionary of metrics.
        """
        assert pairs.shape[-4] > 1, f"Number of pairs should be greater than 1, got {pairs.shape[-4]}."
        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, training)

        if latents_logvar is not None:
            latents, prior_kl_loss, kl_metrics = self._sample_latents(latents_mu, latents_logvar)
            # Compute Gaussian KL between all the latents from each batch.
            pairwise_kl_loss = self._compute_pairwise_gaussian_kl(latents_mu, latents_logvar).mean()
            kl_metrics["pairwise_kl"] = pairwise_kl_loss
        else:
            latents, prior_kl_loss, pairwise_kl_loss, kl_metrics = latents_mu, None, None, {}

        if mode_kwargs.get("remove_encoder_latents", False):
            latents = torch.randn_like(latents)
            
        leave_one_out_latents = make_leave_one_out(latents, axis=-2)  # (*B, N, N-1, H)
        
        if mode == "mean":
            # Compute the context vector by taking the mean of all but one latents.
            context = leave_one_out_latents.mean(dim=-2)  # (*B, N, H)
            # Compute the loss for each pair using the mean of all but one latents.
            loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, training)
        elif mode == "all":
            # Compute the loss for each pair using all but one latents.
            loss_list = []
            metrics_list = []
            for i in range(leave_one_out_latents.shape[-2]):
                l, m = self._loss_from_pair_and_context(
                    leave_one_out_latents[..., i, :], pairs, grid_shapes, training
                )
                loss_list.append(l)
                metrics_list.append(m)
            loss = torch.stack(loss_list, dim=-1)
            metrics = {k: torch.stack([m[k] for m in metrics_list], dim=-1) for k in metrics_list[0].keys()}
            # For logging purposes
            context = latents
            distance_context_latents = torch.norm(
                latents.unsqueeze(-2) - leave_one_out_latents, dim=-1
            )
        elif mode == "random_search":
            for arg in ["num_samples", "scale"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'random_search' training mode."
            
            # Repeat all the pairs and grid shapes except the one to leave out.
            leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)  # (*B, N, N-1, R, C, 2)
            leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)  # (*B, N, N-1, 2, 2)
            # Get the best context for each pair using random search.
            context, _ = self._get_random_search_context(
                leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, **mode_kwargs
            )  # (*B, N, H)
            # Compute the loss for each pair using the context from the random search.
            loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, training)
        elif mode == "gradient_ascent":
            for arg in ["num_steps", "lr"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'gradient_ascent' training mode."
            
            # Repeat all the pairs and grid shapes except the one to leave out.
            leave_one_out_pairs = make_leave_one_out(pairs, axis=-4)  # (*B, N, N-1, R, C, 2)
            leave_one_out_grid_shapes = make_leave_one_out(grid_shapes, axis=-3)  # (*B, N, N-1, 2, 2)
            # Get the best context for each pair using gradient ascent.
            context, _ = self._get_gradient_ascent_context(
                leave_one_out_latents, leave_one_out_pairs, leave_one_out_grid_shapes, **mode_kwargs
            )  # (*B, N, H)
            # Compute the loss for each pair using the context from the gradient ascent.
            loss, metrics = self._loss_from_pair_and_context(context, pairs, grid_shapes, training)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        leave_one_out_contexts = make_leave_one_out(context, axis=-2)
        cosine_between_contexts = F.cosine_similarity(
            context.unsqueeze(-2), leave_one_out_contexts, dim=-1
        )
        cosine_between_latents = F.cosine_similarity(
            latents.unsqueeze(-2), leave_one_out_latents, dim=-1
        )
        
        if mode != "all":
            distance_context_latents = torch.norm(context - latents, dim=-1)
            
        metrics.update({
            "latents_norm": torch.norm(latents, dim=-1),
            "context_norm": torch.norm(context, dim=-1),
            "distance_context_latents": distance_context_latents,
            "distance_between_contexts": torch.norm(
                context.unsqueeze(-2) - leave_one_out_contexts, dim=-1
            ),
            "cosine_between_contexts": cosine_between_contexts,
            "distance_between_latents": torch.norm(
                latents.unsqueeze(-2) - leave_one_out_latents, dim=-1
            ),
            "cosine_between_latents": cosine_between_latents,
        })
        
        # Apply mean to all metrics and loss
        loss = loss.mean()
        metrics = {k: v.mean() for k, v in metrics.items()}
        metrics.update(kl_metrics)
        
        if prior_kl_loss is not None:
            if prior_kl_coeff is None:
                raise ValueError("Prior KL coefficient is required when using variational inference.")
            loss += prior_kl_coeff * prior_kl_loss
            if pairwise_kl_coeff is not None:
                loss += pairwise_kl_coeff * pairwise_kl_loss

        return loss, metrics

    @staticmethod
    def _compute_pairwise_gaussian_kl(mu: torch.Tensor, log_var: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Compute pairwise KL divergence between Gaussian distributions.

        Args:
            mu: mean of shape (*B, N, H)
            log_var: log variance of shape (*B, N, H)

        Returns:
            Mean KL divergence of shape (*B,)
        """
        # Expand dimensions for broadcasting
        mu1 = mu.unsqueeze(-2)  # (*B, N, 1, H)
        mu2 = mu.unsqueeze(-3)  # (*B, 1, N, H)
        log_var1 = log_var.unsqueeze(-2)  # (*B, N, 1, H)
        log_var2 = log_var.unsqueeze(-3)  # (*B, 1, N, H)
        
        # KL divergence formula for Gaussians
        var1, var2 = torch.exp(log_var1), torch.exp(log_var2)
        log_var_ratio = log_var2 - log_var1
        var_ratio = var1 / (var2 + eps)
        mu_diff_sq = (mu1 - mu2) ** 2 / (var2 + eps)
        kl = 0.5 * (log_var_ratio + var_ratio + mu_diff_sq - 1).sum(dim=-1)  # (*B, N, N)
        
        # Average over the pairwise matrices
        num_pairs = mu.shape[-2]
        eye_mask = torch.eye(num_pairs, device=mu.device, dtype=torch.bool)
        kl = torch.where(~eye_mask, kl, torch.zeros_like(kl))
        kl = kl.sum(dim=(-1, -2)) / (num_pairs * (num_pairs - 1))
        return kl

    @staticmethod
    def _sample_latents(
        latents_mu: torch.Tensor, latents_logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        latents_std = torch.exp(0.5 * latents_logvar)
        latents = latents_mu + latents_std * torch.randn_like(latents_mu)
        kl_loss = (-0.5 * (1 + latents_logvar - latents_mu**2 - torch.exp(latents_logvar)).sum(dim=-1)).mean()
        kl_metrics = {
            "prior_kl": kl_loss,
            "latents_mu": latents_mu.mean(),
            "norm_latents_mu": torch.norm(latents_mu, dim=-1).mean(),
            "latents_logvar": latents_logvar.mean(),
        }
        return latents, kl_loss, kl_metrics

    def _loss_from_pair_and_context(
        self,
        context: torch.Tensor,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the loss for a single pair given a context.
        """
        config = self.decoder.config

        # Make the input and output sequences.
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)

        # Decode the output sequence (teacher forcing).
        row_logits, col_logits, grid_logits = self.decoder(input_seq, output_seq, context, training)

        # Compute cross entropy losses.
        grid_shapes_row, grid_shapes_col = grid_shapes[..., 0, 1], grid_shapes[..., 1, 1]
        
        row_loss = F.cross_entropy(row_logits, grid_shapes_row - 1, reduction='none')
        col_loss = F.cross_entropy(col_logits, grid_shapes_col - 1, reduction='none')

        # Copy the grid logits from the last non-padded column of each row to the first column of the next row
        last_non_padded_logits = self._get_last_non_padded_logits(
            grid_logits, grid_shapes_col.unsqueeze(-1).unsqueeze(-1)
        )
        grid_logits_modified = grid_logits.clone()
        max_cols = config.max_cols
        for i in range(1, config.max_rows):
            grid_logits_modified[..., i * max_cols, :] = last_non_padded_logits[..., i-1, :]

        output_labels = pairs[..., 1].reshape(*pairs.shape[:-3], -1)
        grid_losses = F.cross_entropy(
            grid_logits_modified.view(-1, config.output_vocab_size),
            output_labels.view(-1),
            reduction='none'
        ).view(output_labels.shape)
        
        grid_loss = self._normalized_mean_over_sequence(grid_losses, grid_shapes_row, grid_shapes_col)

        loss = row_loss + col_loss + grid_loss
        metrics = {
            "shape_row_loss": row_loss,
            "shape_col_loss": col_loss,
            "grid_loss": grid_loss,
            "total_loss": loss,
        }
        return loss, metrics

    def _normalized_mean_over_sequence(
        self, grid_seq: torch.Tensor, num_rows: torch.Tensor, num_cols: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the mean of the sequence over the valid (non-padded) positions.
        """
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        device = grid_seq.device
        
        row_arange = torch.arange(max_rows, device=device).view(
            *([1] * len(num_rows.shape)), max_rows
        )
        col_arange = torch.arange(max_cols, device=device).view(
            *([1] * len(num_cols.shape)), max_cols
        )
        
        grid_row_mask = row_arange < num_rows.unsqueeze(-1)
        grid_col_mask = col_arange < num_cols.unsqueeze(-1)
        grid_pad_mask = grid_row_mask.unsqueeze(-1) & grid_col_mask.unsqueeze(-2)
        grid_pad_mask = grid_pad_mask.view(*grid_pad_mask.shape[:-2], -1)
        
        # Mask the elements corresponding to the padding tokens.
        grid_seq = torch.where(grid_pad_mask, grid_seq, torch.zeros_like(grid_seq))
        # Mean over the sequence length, normalizing by the number of non-padded tokens.
        mean_seq = grid_seq.sum(dim=-1) / (grid_pad_mask.sum(dim=-1) + 1e-5)
        return mean_seq

    def generate_output(
        self,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        input: torch.Tensor,
        input_grid_shape: torch.Tensor,
        training: bool = False,
        mode: Literal["mean", "first", "random_search", "gradient_ascent"] = "mean",
        return_two_best: bool = False,
        **mode_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predicts the output grid given an input grid and other (input, output) pairs.
        """
        latents_mu, latents_logvar = self.encoder(pairs, grid_shapes, training)

        if latents_logvar is not None:
            latents, *_ = self._sample_latents(latents_mu, latents_logvar)
        else:
            latents = latents_mu

        if mode_kwargs.get("remove_encoder_latents", False):
            latents = torch.randn_like(latents)
            
        if mode == "mean":
            context = latents.mean(dim=-2)
            first_context, second_context = context, context
        elif mode == "first":
            context = latents[..., 0, :]
            first_context, second_context = context, context
        elif mode == "random_search":
            for arg in ["num_samples", "scale"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'random_search' inference mode."

            first_context, second_context = self._get_random_search_context(
                latents, pairs, grid_shapes, **mode_kwargs
            )
        elif mode == "gradient_ascent":
            for arg in ["num_steps", "lr"]:
                assert arg in mode_kwargs, f"'{arg}' argument required for 'gradient_ascent' inference mode."

            first_context, second_context = self._get_gradient_ascent_context(
                latents, pairs, grid_shapes, **mode_kwargs
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        info = {"context": first_context}

        if return_two_best:
            first_output_grids, first_output_shapes = self._generate_output_from_context(
                first_context, input, input_grid_shape, training
            )
            second_output_grids, second_output_shapes = self._generate_output_from_context(
                second_context, input, input_grid_shape, training
            )
            return first_output_grids, first_output_shapes, second_output_grids, second_output_shapes, info
        else:
            output_grids, output_shapes = self._generate_output_from_context(
                first_context, input, input_grid_shape, training
            )
            return output_grids, output_shapes, info

    def _generate_output_from_context(
        self, context: torch.Tensor, input: torch.Tensor, input_grid_shape: torch.Tensor, training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flattened_input = input.view(*input.shape[:-2], -1)
        input_seq = torch.cat([input_grid_shape, flattened_input], dim=-1)
        output_seq = torch.zeros_like(input_seq)
        output_seq[..., :2] = 1  # Initialize the grid shape tokens to 1.

        def grid_shape_step(output_seq: torch.Tensor, row: bool) -> torch.Tensor:
            row_logits, col_logits, _ = self.decoder(input_seq, output_seq, context, training)
            if row:
                logits = row_logits
            else:
                logits = col_logits
            # +1 to shift the tokens to [1, max_rows] or [1, max_cols]
            new_token = torch.argmax(logits, dim=-1) + 1
            output_seq = output_seq.clone()
            output_seq[..., int(not row)] = new_token
            return output_seq

        # First predict the number of rows and then the number of columns.
        output_seq = grid_shape_step(output_seq, row=True)
        output_seq = grid_shape_step(output_seq, row=False)
        output_shapes = output_seq[..., :2]
        max_cols = self.decoder.config.max_cols

        # Then predict the grid values.
        for i in range(self.decoder.config.max_len):
            *_, grid_logits = self.decoder(input_seq, output_seq, context, training)
            # If we are at the beginning of a new row, the index of the logits to predict the next token
            logits_index = torch.where(
                (i % max_cols == 0) & (i > 0),
                (i // max_cols - 1) * max_cols + output_shapes[..., 1],
                torch.tensor(i, device=output_seq.device)
            )
            logits = grid_logits.gather(-2, logits_index.unsqueeze(-1).unsqueeze(-1).expand(
                *logits_index.shape, grid_logits.shape[-1]
            )).squeeze(-2)
            new_token = torch.argmax(logits, dim=-1)
            output_seq = output_seq.clone()
            output_seq[..., 2 + i] = new_token  # +2 to skip the grid shapes
        
        output_grids = output_seq[..., 2:].view(*input.shape[:-2], *input.shape[-2:])
        return output_grids, output_shapes

    def _get_random_search_context(
        self,
        latents: torch.Tensor,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        num_samples: int,
        scale: float,
        scan_batch_size: Optional[int] = None,
        include_mean_latent: bool = True,
        include_all_latents: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the best two contexts using a batched random search."""
        latents = self._prepare_latents_before_search(include_mean_latent, include_all_latents, latents)

        if num_samples > 0:
            # Sample some random latents around the given latents.
            num_latents = latents.shape[-2]
            num_padded_samples = math.ceil(num_samples / num_latents) * num_latents
            random_vectors = torch.randn(
                *latents.shape[:-2], num_latents, num_padded_samples // num_latents, latents.shape[-1],
                device=latents.device
            )
            random_latents = latents.unsqueeze(-2) + scale * random_vectors
            random_latents = random_latents.view(*random_latents.shape[:-3], -1, random_latents.shape[-1])
            random_latents = random_latents[..., :num_samples, :]
            latents = torch.cat([latents, random_latents], dim=-2)

        # Flatten input/output for decoding likelihood
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)

        # Use the same latent for all pairs of the same task.
        latents = latents.unsqueeze(-3).repeat(*([1] * (latents.ndim - 2)), output_seq.shape[-2], 1, 1)

        # Compute log probabilities for all latents
        log_probs_list = []
        for i in range(latents.shape[-2]):
            row_logits, col_logits, grid_logits = self.decoder(
                input_seq, output_seq, latents[..., i, :], training=False
            )
            log_probs = self._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)
            log_probs_list.append(log_probs)
        
        log_probs = torch.stack(log_probs_list, dim=-1)

        # Remove the duplication of the latents over the pairs.
        latents = latents[..., 0, :, :]
        best_context, second_best_context = self._select_best_and_second_best_latents(log_probs, latents)

        return best_context, second_best_context

    def _get_gradient_ascent_context(
        self,
        latents: torch.Tensor,
        pairs: torch.Tensor,
        grid_shapes: torch.Tensor,
        num_steps: int,
        lr: float,
        lr_schedule: bool = False,
        lr_schedule_exponent: float = 0.5,
        optimizer: Literal["sgd", "adam"] = "sgd",
        optimizer_kwargs: Optional[dict] = None,
        include_mean_latent: bool = True,
        include_all_latents: bool = False,
        random_perturbation: Optional[dict] = None,
        stop_gradient_latent_move: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the best two contexts using a gradient ascent algorithm."""
        latents = self._prepare_latents_before_search(
            include_mean_latent, include_all_latents, latents, random_perturbation
        )

        # Flatten input/output for decoding likelihood
        input_seq, output_seq = self._flatten_input_output_for_decoding(pairs, grid_shapes)

        # Make latents require gradients
        latents = latents.clone().requires_grad_(True)

        # Setup optimizer
        if optimizer == "sgd":
            opt = optim.SGD([latents], lr=lr, **(optimizer_kwargs or {}))
        elif optimizer == "adam":
            opt = optim.Adam([latents], lr=lr, **(optimizer_kwargs or {}))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        if lr_schedule:
            scheduler = CosineAnnealingLR(opt, T_max=num_steps)

        all_latents = [latents.clone().detach()]
        all_log_probs = []

        for step in range(num_steps):
            opt.zero_grad()
            
            # Use the same latent for all pairs of the same task.
            expanded_latents = latents.unsqueeze(-3).repeat(
                *([1] * (latents.ndim - 2)), output_seq.shape[-2], 1, 1
            )
            
            row_logits, col_logits, grid_logits = self.decoder(
                input_seq, output_seq, expanded_latents, training=False
            )
            log_probs = self._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)
            
            # Maximize log probability (minimize negative log probability)
            loss = -log_probs.sum()
            loss.backward()
            
            if stop_gradient_latent_move:
                # Don't propagate gradients through latent modifications
                with torch.no_grad():
                    opt.step()
                    if lr_schedule:
                        scheduler.step()
            else:
                opt.step()
                if lr_schedule:
                    scheduler.step()
            
            all_latents.append(latents.clone().detach())
            all_log_probs.append(log_probs.detach())

        # Compute final log probabilities
        with torch.no_grad():
            expanded_latents = latents.unsqueeze(-3).repeat(
                *([1] * (latents.ndim - 2)), output_seq.shape[-2], 1, 1
            )
            row_logits, col_logits, grid_logits = self.decoder(
                input_seq, output_seq, expanded_latents, training=False
            )
            final_log_probs = self._compute_log_probs(row_logits, col_logits, grid_logits, output_seq)

        # Concatenate all latents and log probs
        all_latents = torch.stack(all_latents, dim=-2)
        all_log_probs.append(final_log_probs)
        all_log_probs = torch.stack(all_log_probs, dim=-1)
        
        # Flatten latents and log probs
        latents_flat = all_latents.view(*all_latents.shape[:-2], -1, all_latents.shape[-1])
        log_probs_flat = all_log_probs.view(*all_log_probs.shape[:-1], -1)

        best_context, second_best_context = self._select_best_and_second_best_latents(
            log_probs_flat, latents_flat
        )

        return best_context, second_best_context

    @classmethod
    def _prepare_latents_before_search(
        cls,
        include_mean_latent: bool,
        include_all_latents: bool,
        latents: torch.Tensor,
        random_perturbation: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Selects the latents from which to start the search.
        """
        if include_mean_latent:
            mean_latent = latents.mean(dim=-2, keepdim=True)
            if include_all_latents:
                # Include the mean latent in the latents from which to start the search.
                prep_latents = torch.cat([mean_latent, latents], dim=-2)
            else:
                # Only start the search from the mean latent.
                prep_latents = mean_latent
        else:
            # Start the search from all the pair latents.
            if not include_all_latents:
                raise ValueError(
                    "At least one of 'include_mean_latent' or 'include_all_latents' should be True."
                )
            prep_latents = latents
            
        if random_perturbation is not None:
            for arg in ["num_samples", "scale"]:
                assert arg in random_perturbation, f"'{arg}' argument required for random perturbation."
            num_samples = random_perturbation["num_samples"]
            scale = random_perturbation["scale"]
            random_vectors = torch.randn(
                *latents.shape[:-2], num_samples, latents.shape[-1], device=latents.device
            )
            random_latents = latents.mean(dim=-2, keepdim=True) + scale * random_vectors
            prep_latents = torch.cat([prep_latents, random_latents], dim=-2)
        return prep_latents

    @classmethod
    def _flatten_input_output_for_decoding(
        cls, pairs: torch.Tensor, grid_shapes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flattened_pairs = pairs.view(*pairs.shape[:-3], -1, 2)
        input_seq = torch.cat([grid_shapes[..., 0], flattened_pairs[..., 0]], dim=-1)
        output_seq = torch.cat([grid_shapes[..., 1], flattened_pairs[..., 1]], dim=-1)
        return input_seq, output_seq

    @classmethod
    def _select_best_and_second_best_latents(
        cls, log_probs: torch.Tensor, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sorted_indices = torch.argsort(log_probs, dim=-1, descending=True)
        best_context = torch.gather(
            latents, -2, sorted_indices[..., 0:1].unsqueeze(-1).expand(-1, -1, latents.shape[-1])
        ).squeeze(-2)
        
        try:
            second_best_context = torch.gather(
                latents, -2, sorted_indices[..., 1:2].unsqueeze(-1).expand(-1, -1, latents.shape[-1])
            ).squeeze(-2)
        except (IndexError, RuntimeError):
            # If there is only one latent, the second best context is the same as the best context.
            second_best_context = best_context
        return best_context, second_best_context

    def _compute_log_probs(
        self,
        row_logits: torch.Tensor,
        col_logits: torch.Tensor,
        grid_logits: torch.Tensor,
        output_seq: torch.Tensor,
        grid_log_prob_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Computes the log probabilities of the given output sequence.
        """
        max_cols = self.decoder.config.max_cols
        num_rows, num_cols = output_seq[..., 0], output_seq[..., 1]
        
        row_log_probs = F.log_softmax(row_logits, dim=-1)
        row_log_prob = torch.gather(row_log_probs, -1, (num_rows - 1).unsqueeze(-1)).squeeze(-1)
        
        col_log_probs = F.log_softmax(col_logits, dim=-1)
        col_log_prob = torch.gather(col_log_probs, -1, (num_cols - 1).unsqueeze(-1)).squeeze(-1)
        
        # Copy the grid logits from the last non-padded column of each row to the first column of the next row
        last_non_padded_logits = self._get_last_non_padded_logits(
            grid_logits, num_cols.unsqueeze(-1).unsqueeze(-1)
        )
        grid_logits_modified = grid_logits.clone()
        for i in range(1, self.decoder.config.max_rows):
            grid_logits_modified[..., i * max_cols, :] = last_non_padded_logits[..., i-1, :]

        grid_log_probs = F.log_softmax(grid_logits_modified, dim=-1)
        grid_log_prob = torch.gather(
            grid_log_probs, -1, output_seq[..., 2:].unsqueeze(-1)
        ).squeeze(-1)
        grid_log_prob = self._normalized_mean_over_sequence(grid_log_prob, num_rows, num_cols)

        log_probs = row_log_prob + col_log_prob + grid_log_prob_weight * grid_log_prob
        log_probs = log_probs.sum(dim=-1)  # sum log_probs over the pairs
        return log_probs

    def _get_last_non_padded_logits(self, grid_logits: torch.Tensor, num_cols: torch.Tensor) -> torch.Tensor:
        """Selects the grid logits from the last non-padded column of each row."""
        max_rows, max_cols = self.decoder.config.max_rows, self.decoder.config.max_cols
        last_non_padded_logits = []
        
        for i in range(1, max_rows):
            indices = max_cols * i - (max_cols - num_cols.squeeze(-1).squeeze(-1))
            end_of_row_logits = torch.gather(
                grid_logits, -2, indices.unsqueeze(-1).unsqueeze(-1).expand(
                    *indices.shape, grid_logits.shape[-1]
                )
            )
            last_non_padded_logits.append(end_of_row_logits)
        
        return torch.cat(last_non_padded_logits, dim=-2)


if __name__ == "__main__":
    # Mock config classes for testing
    class TransformerLayerConfig:
        def __init__(self, dropout_rate=0.05):
            self.dropout_rate = dropout_rate
            self.use_bias = True

    class EncoderTransformerConfig:
        def __init__(self, vocab_size=10, max_rows=5, max_cols=5, latent_dim=64, variational=True):
            self.vocab_size = vocab_size
            self.max_rows = max_rows
            self.max_cols = max_cols
            self.latent_dim = latent_dim
            self.variational = variational
            self.transformer_layer = TransformerLayerConfig()

    class DecoderTransformerConfig:
        def __init__(self, vocab_size=10, max_rows=5, max_cols=5, output_vocab_size=10):
            self.vocab_size = vocab_size
            self.output_vocab_size = output_vocab_size
            self.max_rows = max_rows
            self.max_cols = max_cols
            self.max_len = max_rows * max_cols
            self.transformer_layer = TransformerLayerConfig()

    batch_size = 4
    mini_batch_size = 3
    max_rows = 5
    max_cols = 5
    vocab_size = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_config = EncoderTransformerConfig(
        vocab_size=vocab_size,
        max_rows=max_rows,
        max_cols=max_cols,
        variational=True,
    )
    decoder_config = DecoderTransformerConfig(
        vocab_size=vocab_size,
        max_rows=max_rows,
        max_cols=max_cols,
        output_vocab_size=vocab_size,
    )

    encoder = EncoderTransformer(encoder_config).to(device)
    decoder = DecoderTransformer(decoder_config).to(device)
    lpn = LPN(encoder=encoder, decoder=decoder).to(device)

    pairs = torch.randint(
        0, vocab_size,
        (batch_size, mini_batch_size, max_rows, max_cols, 2),
        device=device
    )
    grid_shapes = torch.randint(
        1, min(max_rows, max_cols) + 1,
        (batch_size, mini_batch_size, 2, 2),
        device=device
    )
    
    num_parameters = sum(p.numel() for p in lpn.parameters())
    print(f"Number of parameters: {num_parameters:,}")

    with torch.no_grad():
        loss, metrics = lpn(
            pairs, grid_shapes, training=False, mode="mean", 
            prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4
        )
        print("Mean Loss:", loss.item())

        loss, metrics = lpn(
            pairs, grid_shapes, training=False, mode="all",
            prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4
        )
        print("All Loss:", loss.item())

        loss, metrics = lpn(
            pairs, grid_shapes, training=False, mode="random_search",
            prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4,
            num_samples=10, scale=1.0
        )
        print("Random Search Loss:", loss.item())

        loss, metrics = lpn(
            pairs, grid_shapes, training=False, mode="gradient_ascent",
            prior_kl_coeff=1e-4, pairwise_kl_coeff=1e-4,
            num_steps=2, lr=5e-2, optimizer="adam",
            optimizer_kwargs={"betas": (0.5, 0.999)},
            include_all_latents=True,
            random_perturbation={"num_samples": 3, "scale": 0.5}
        )
        print("Gradient Ascent Loss:", loss.item())

        output_grids, output_shapes, info = lpn.generate_output(
            pairs, grid_shapes, pairs[:, 0, ..., 0], grid_shapes[:, 0, ..., 0],
            training=False, mode="random_search",
            num_samples=10, scale=2.0,
            include_mean_latent=True, include_all_latents=False
        )
        print("Random search")
        print("Output grids of shape:", output_grids.shape)
        print("Output shapes of shape:", output_shapes.shape)

        output_grids, output_shapes, info = lpn.generate_output(
            pairs, grid_shapes, pairs[:, 0, ..., 0], grid_shapes[:, 0, ..., 0],
            training=False, mode="gradient_ascent",
            num_steps=10, lr=5e-3, optimizer="adam",
            optimizer_kwargs={"betas": (0.5, 0.999)},
            include_mean_latent=True, include_all_latents=False,
            random_perturbation={"num_samples": 3, "scale": 0.5},
            remove_encoder_latents=True
        )
        print("Gradient ascent")
        print("Output grids of shape:", output_grids.shape)
        print("Output shapes of shape:", output_shapes.shape)