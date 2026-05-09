"""Attention Perturbation Sensitivity Network for idea i106.

A small attention reader produces per-query attention maps over the 64 board
squares and a base latent.  Four deterministic mask families
(top-attention, low-attention occupied, permutation-random occupied, and the
3x3 neighbourhood of the top-attention square) zero the 12 piece planes at the
selected squares.  The shared encoder is re-run on each masked variant and the
puzzle classifier reads the base latent together with sensitivity contrasts
``||z(x) - z(mask_*(x))||`` and a small set of attention diagnostics.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


PIECE_PLANES = 12


def _board_coords() -> torch.Tensor:
    rank = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 8)
    file = torch.arange(8, dtype=torch.float32).view(1, 8).expand(8, 8)
    centered_rank = (rank - 3.5) / 3.5
    centered_file = (file - 3.5) / 3.5
    edge_distance = torch.minimum(
        torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)
    ) / 3.5
    square_color = ((rank + file).remainder(2.0) * 2.0) - 1.0
    return torch.stack(
        [
            rank / 7.0,
            file / 7.0,
            centered_rank,
            centered_file,
            edge_distance,
            square_color,
        ],
        dim=-1,
    ).view(64, 6)


def _neighborhood_3x3() -> torch.Tensor:
    adj = torch.zeros(64, 64, dtype=torch.float32)
    for r in range(8):
        for f in range(8):
            s = r * 8 + f
            for dr in (-1, 0, 1):
                for df in (-1, 0, 1):
                    nr, nf = r + dr, f + df
                    if 0 <= nr < 8 and 0 <= nf < 8:
                        adj[s, nr * 8 + nf] = 1.0
    return adj


def _deterministic_permutation(seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(64, generator=generator)


class _SquareEncoder(nn.Module):
    """Tokenises 64 squares from board planes + deterministic coords."""

    def __init__(self, input_channels: int, token_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.register_buffer("coords", _board_coords(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        per_square = x.flatten(2).transpose(1, 2)
        coords = self.coords.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(b, -1, -1)
        return self.mlp(torch.cat([per_square, coords], dim=-1))


class _AttentionReader(nn.Module):
    """Q learnable queries -> attention over 64 square tokens + base latent."""

    def __init__(
        self,
        input_channels: int,
        token_dim: int,
        hidden_dim: int,
        num_queries: int,
    ) -> None:
        super().__init__()
        self.tokenizer = _SquareEncoder(input_channels, token_dim, hidden_dim)
        self.queries = nn.Parameter(torch.empty(num_queries, token_dim))
        nn.init.xavier_uniform_(self.queries)
        self.to_k = nn.Linear(token_dim, token_dim, bias=False)
        self.to_v = nn.Linear(token_dim, token_dim, bias=False)
        self.latent_norm = nn.LayerNorm(token_dim)
        self.token_dim = int(token_dim)
        self.num_queries = int(num_queries)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(x)
        keys = self.to_k(tokens)
        values = self.to_v(tokens)
        scale = 1.0 / math.sqrt(float(self.token_dim))
        scores = torch.matmul(self.queries, keys.transpose(-1, -2)) * scale
        attention = F.softmax(scores, dim=-1)
        per_query = torch.matmul(attention, values)
        latent = self.latent_norm(per_query.mean(dim=1))
        return latent, attention


class AttentionPerturbationSensitivityNetwork(nn.Module):
    """Bespoke attention-guided perturbation-sensitivity bottleneck."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 64,
        hidden_dim: int = 96,
        head_hidden: int = 128,
        num_queries: int = 8,
        top_k: int = 6,
        dropout: float = 0.1,
        permutation_seed: int = 1064,
        sensitivity_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "AttentionPerturbationSensitivityNetwork supports the puzzle_binary one-logit contract"
            )
        if input_channels < PIECE_PLANES + 1:
            raise ValueError("input_channels must be at least 12 piece planes plus globals")
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        if not 1 <= top_k <= 64:
            raise ValueError("top_k must be in [1, 64]")

        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.input_channels = int(input_channels)
        self.token_dim = int(token_dim)
        self.num_queries = int(num_queries)
        self.top_k = int(top_k)
        self.sensitivity_eps = float(sensitivity_eps)

        self.encoder = _AttentionReader(
            input_channels=int(input_channels),
            token_dim=self.token_dim,
            hidden_dim=int(hidden_dim),
            num_queries=self.num_queries,
        )

        self.register_buffer("neighborhood", _neighborhood_3x3(), persistent=False)
        permutation = _deterministic_permutation(int(permutation_seed))
        position = torch.empty(64, dtype=torch.long)
        position[permutation] = torch.arange(64)
        self.register_buffer("permutation", permutation, persistent=False)
        self.register_buffer("permutation_position", position.float(), persistent=False)

        # 1 latent (D) + 4 sensitivity scalars + 4 contrasts + 7 attention diagnostics
        diagnostic_dim = 4 + 4 + 7
        head_input = self.token_dim + diagnostic_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), 1),
        )

    def _occupancy(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :PIECE_PLANES].sum(dim=1) > 0).flatten(1).to(dtype=x.dtype)

    def _aggregate_attention(self, attention: torch.Tensor) -> torch.Tensor:
        return attention.mean(dim=1)

    def _topk_keep_mask(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        _, idx = torch.topk(scores, k, dim=-1)
        keep = torch.ones_like(scores)
        keep.scatter_(-1, idx, 0.0)
        return keep

    def _low_attention_occupied_keep_mask(
        self, scores: torch.Tensor, occupancy: torch.Tensor, k: int
    ) -> torch.Tensor:
        large = torch.finfo(scores.dtype).max / 4.0
        masked = torch.where(occupancy > 0.5, scores, torch.full_like(scores, large))
        # ask for the K smallest among occupied squares
        _, idx = torch.topk(masked, k, dim=-1, largest=False)
        keep = torch.ones_like(scores)
        keep.scatter_(-1, idx, 0.0)
        return keep

    def _random_occupied_keep_mask(
        self, occupancy: torch.Tensor, k: int
    ) -> torch.Tensor:
        position = self.permutation_position.to(dtype=occupancy.dtype, device=occupancy.device)
        scores = position.unsqueeze(0).expand_as(occupancy)
        large = torch.finfo(scores.dtype).max / 4.0
        masked = torch.where(occupancy > 0.5, scores, torch.full_like(scores, large))
        _, idx = torch.topk(masked, k, dim=-1, largest=False)
        keep = torch.ones_like(occupancy)
        keep.scatter_(-1, idx, 0.0)
        return keep

    def _top_neighborhood_keep_mask(self, scores: torch.Tensor) -> torch.Tensor:
        top_idx = scores.argmax(dim=-1)
        adjacency = self.neighborhood.to(dtype=scores.dtype, device=scores.device)
        nbhd = adjacency[top_idx]
        return 1.0 - nbhd

    def _apply_keep_mask(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)
        keep_mask = keep.unsqueeze(1)
        piece = flat[:, :PIECE_PLANES] * keep_mask
        rest = flat[:, PIECE_PLANES:]
        masked = torch.cat([piece, rest], dim=1)
        return masked.reshape(b, c, h, w)

    def _attention_diagnostics(
        self, attention: torch.Tensor, occupancy: torch.Tensor, top_k: int
    ) -> dict[str, torch.Tensor]:
        eps = 1.0e-9
        per_query_entropy = -(attention.clamp_min(eps).log() * attention).sum(dim=-1)
        mean_query_entropy = per_query_entropy.mean(dim=-1) / math.log(64.0)
        max_attention = attention.amax(dim=(-1, -2))
        topk_values, _ = torch.topk(attention, top_k, dim=-1)
        topk_mass = topk_values.sum(dim=-1).mean(dim=-1)
        per_square = self._aggregate_attention(attention)
        occupied_mass = (per_square * occupancy).sum(dim=-1)
        empty_mass = (per_square * (1.0 - occupancy)).sum(dim=-1)
        attention_var_across_queries = attention.var(dim=1, unbiased=False).sum(dim=-1)
        per_square_max_minus_min = per_square.amax(dim=-1) - per_square.amin(dim=-1)
        return {
            "attention": attention,
            "per_square_attention": per_square,
            "mean_query_entropy": mean_query_entropy,
            "max_attention": max_attention,
            "topk_attention_mass": topk_mass,
            "attention_occupied_mass": occupied_mass,
            "attention_empty_mass": empty_mass,
            "attention_query_disagreement": attention_var_across_queries,
            "attention_range": per_square_max_minus_min,
        }

    def _sensitivity(
        self, base: torch.Tensor, perturbed: torch.Tensor
    ) -> torch.Tensor:
        return (base - perturbed).norm(dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        latent_base, attention = self.encoder(x)

        occupancy = self._occupancy(x)
        per_square_scores = self._aggregate_attention(attention).detach()

        keep_top = self._topk_keep_mask(per_square_scores, self.top_k)
        keep_low = self._low_attention_occupied_keep_mask(per_square_scores, occupancy, self.top_k)
        keep_rand = self._random_occupied_keep_mask(occupancy, self.top_k)
        keep_nbhd = self._top_neighborhood_keep_mask(per_square_scores)

        x_top = self._apply_keep_mask(x, keep_top)
        x_low = self._apply_keep_mask(x, keep_low)
        x_rand = self._apply_keep_mask(x, keep_rand)
        x_nbhd = self._apply_keep_mask(x, keep_nbhd)

        latent_top, _ = self.encoder(x_top)
        latent_low, _ = self.encoder(x_low)
        latent_rand, _ = self.encoder(x_rand)
        latent_nbhd, _ = self.encoder(x_nbhd)

        delta_top = self._sensitivity(latent_base, latent_top)
        delta_low = self._sensitivity(latent_base, latent_low)
        delta_rand = self._sensitivity(latent_base, latent_rand)
        delta_nbhd = self._sensitivity(latent_base, latent_nbhd)

        contrast_top_low = delta_top - delta_low
        contrast_top_rand = delta_top - delta_rand
        contrast_nbhd_top = delta_nbhd - delta_top
        ratio_top_low = delta_top / (delta_low + self.sensitivity_eps)

        diagnostics = self._attention_diagnostics(attention, occupancy, self.top_k)

        sensitivity_features = torch.stack(
            [
                delta_top,
                delta_low,
                delta_rand,
                delta_nbhd,
                contrast_top_low,
                contrast_top_rand,
                contrast_nbhd_top,
                ratio_top_low,
            ],
            dim=-1,
        )
        attention_features = torch.stack(
            [
                diagnostics["mean_query_entropy"],
                diagnostics["max_attention"],
                diagnostics["topk_attention_mass"],
                diagnostics["attention_occupied_mass"],
                diagnostics["attention_empty_mass"],
                diagnostics["attention_query_disagreement"],
                diagnostics["attention_range"],
            ],
            dim=-1,
        )

        features = torch.cat([latent_base, sensitivity_features, attention_features], dim=-1)
        logits = self.classifier(features).view(-1)

        return {
            "logits": logits,
            "latent_base": latent_base,
            "latent_top": latent_top,
            "latent_low": latent_low,
            "latent_random": latent_rand,
            "latent_neighborhood": latent_nbhd,
            "attention": attention,
            "per_square_attention": diagnostics["per_square_attention"],
            "mean_query_entropy": diagnostics["mean_query_entropy"],
            "max_attention": diagnostics["max_attention"],
            "topk_attention_mass": diagnostics["topk_attention_mass"],
            "attention_occupied_mass": diagnostics["attention_occupied_mass"],
            "attention_empty_mass": diagnostics["attention_empty_mass"],
            "attention_query_disagreement": diagnostics["attention_query_disagreement"],
            "attention_range": diagnostics["attention_range"],
            "occupancy_mask": occupancy,
            "keep_mask_top": keep_top,
            "keep_mask_low": keep_low,
            "keep_mask_random": keep_rand,
            "keep_mask_neighborhood": keep_nbhd,
            "delta_top": delta_top,
            "delta_low": delta_low,
            "delta_random": delta_rand,
            "delta_neighborhood": delta_nbhd,
            "contrast_top_minus_low": contrast_top_low,
            "contrast_top_minus_random": contrast_top_rand,
            "contrast_neighborhood_minus_top": contrast_nbhd_top,
            "ratio_top_over_low": ratio_top_low,
            "sensitivity_features": sensitivity_features,
            "attention_features": attention_features,
        }


def build_attention_perturbation_sensitivity_network_from_config(
    config: dict[str, Any],
) -> AttentionPerturbationSensitivityNetwork:
    cfg = dict(config)
    token_dim = int(cfg.get("token_dim", cfg.get("channels", 64)))
    return AttentionPerturbationSensitivityNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=token_dim,
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        head_hidden=int(cfg.get("head_hidden", cfg.get("hidden_dim", 128))),
        num_queries=int(cfg.get("num_queries", 8)),
        top_k=int(cfg.get("top_k", 6)),
        dropout=float(cfg.get("dropout", 0.1)),
        permutation_seed=int(cfg.get("permutation_seed", 1064)),
        sensitivity_eps=float(cfg.get("sensitivity_eps", 1.0e-6)),
    )
