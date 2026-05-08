"""Attention Disagreement Residual Network for idea i103.

Builds square tokens from the board, runs F independent query banks of Q
queries each, and classifies puzzle-likeness from the mean attended value
plus residual-disagreement statistics across families.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


class _SquareTokenizer(nn.Module):
    def __init__(self, input_channels: int, token_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        rank = torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 8)
        file = torch.arange(8, dtype=torch.float32).view(1, 8).expand(8, 8)
        centered_rank = (rank - 3.5) / 3.5
        centered_file = (file - 3.5) / 3.5
        edge_distance = torch.minimum(
            torch.minimum(rank, 7.0 - rank), torch.minimum(file, 7.0 - file)
        ) / 3.5
        square_color = ((rank + file).remainder(2.0) * 2.0) - 1.0
        coords = torch.stack(
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
        self.register_buffer("coords", coords, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_values = x.flatten(2).transpose(1, 2)
        coords = self.coords.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.mlp(torch.cat([square_values, coords], dim=-1))


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    m = 0.5 * (p + q)
    p_log = (p.clamp_min(eps).log() - m.clamp_min(eps).log())
    q_log = (q.clamp_min(eps).log() - m.clamp_min(eps).log())
    return 0.5 * ((p * p_log).sum(dim=-1) + (q * q_log).sum(dim=-1))


class AttentionDisagreementResidualNetwork(nn.Module):
    """Residual disagreement among F independent query families over shared board tokens."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 64,
        family_count: int = 4,
        query_count: int = 8,
        hidden_dim: int = 96,
        head_hidden: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("AttentionDisagreementResidualNetwork supports the puzzle_binary one-logit contract")
        if token_dim < 1 or family_count < 2 or query_count < 1:
            raise ValueError("token_dim, query_count must be positive and family_count >= 2")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.token_dim = int(token_dim)
        self.family_count = int(family_count)
        self.query_count = int(query_count)

        self.tokenizer = _SquareTokenizer(
            input_channels=int(input_channels),
            token_dim=self.token_dim,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )

        query_init = torch.randn(self.family_count, self.query_count, self.token_dim) / math.sqrt(float(self.token_dim))
        self.query_banks = nn.Parameter(query_init)
        self.query_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.key_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.value_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.value_norm = nn.LayerNorm(self.token_dim)
        self.attention_dropout = nn.Dropout(float(attention_dropout)) if attention_dropout > 0 else nn.Identity()

        family_pair_count = self.family_count * (self.family_count - 1) // 2
        # Disagreement features:
        # mean attended value (D), per-family attended std (D),
        # pairwise JS divergence (P), entropy mean per family (F), entropy variance scalar (1),
        # max attended cosine distance (1), mean attended cosine distance (1),
        # attended coord covariance scalar (1), max query-map cosine distance (1).
        diag_dim = (
            2 * self.token_dim
            + family_pair_count
            + self.family_count
            + 4
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(diag_dim),
            nn.Linear(diag_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), 1),
        )
        self._family_pair_count = family_pair_count

    def _attention(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = tokens.shape[0]
        scale = 1.0 / math.sqrt(float(self.token_dim))
        keys = self.key_proj(tokens)
        values = self.value_proj(tokens)
        queries = self.query_proj(self.query_banks.view(-1, self.token_dim)).view(
            self.family_count, self.query_count, self.token_dim
        )
        # scores: (B, F, Q, 64)
        scores = torch.einsum("fqd,bnd->bfqn", queries, keys) * scale
        attention = F.softmax(scores, dim=-1)
        attended = torch.einsum("bfqn,bnd->bfqd", self.attention_dropout(attention), values)
        attended = self.value_norm(attended)
        return attention, attended

    def _disagreement(
        self, attention: torch.Tensor, attended: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, f, q, n = attention.shape

        # Per-query attended pooled across the query axis -> (B, F, D)
        attended_per_family = attended.mean(dim=2)
        attended_mean = attended_per_family.mean(dim=1)  # (B, D)
        attended_residual = attended_per_family - attended_mean.unsqueeze(1)
        attended_std = attended_residual.square().mean(dim=1).clamp_min(0.0).sqrt()  # (B, D)

        # Family-averaged attention map (B, F, 64) for JS divergence
        family_map = attention.mean(dim=2)
        # Pairwise JS divergence
        pair_js = []
        for i in range(self.family_count):
            for j in range(i + 1, self.family_count):
                pair_js.append(_js_divergence(family_map[:, i], family_map[:, j]))
        if pair_js:
            js_pairs = torch.stack(pair_js, dim=-1)  # (B, P)
        else:
            js_pairs = attention.new_zeros(b, 0)
        js_mean = (
            js_pairs.mean(dim=-1)
            if js_pairs.shape[-1] > 0
            else attention.new_zeros(b)
        )
        js_max = (
            js_pairs.amax(dim=-1)
            if js_pairs.shape[-1] > 0
            else attention.new_zeros(b)
        )

        # Entropy per (family, query) -> normalized to [0, 1] by log(64)
        entropy = -(attention * attention.clamp_min(1.0e-8).log()).sum(dim=-1) / math.log(64.0)
        entropy_per_family = entropy.mean(dim=2)  # (B, F)
        entropy_variance = entropy_per_family.var(dim=1, unbiased=False)  # (B,)

        # Cosine distances between family-averaged attention maps
        family_map_norm = F.normalize(family_map, p=2, dim=-1, eps=1.0e-8)
        cos_matrix = torch.einsum("bfd,bgd->bfg", family_map_norm, family_map_norm)
        eye = torch.eye(self.family_count, dtype=cos_matrix.dtype, device=cos_matrix.device).unsqueeze(0)
        cos_distance = (1.0 - cos_matrix) * (1.0 - eye)
        offdiag_count = max(1, self.family_count * (self.family_count - 1))
        cos_distance_mean = cos_distance.sum(dim=(1, 2)) / float(offdiag_count)
        cos_distance_max = cos_distance.amax(dim=(1, 2))

        # Per-query (within family) cosine distance of attention maps to find max query-map disagreement
        # Reshape attention to (B, F*Q, 64) for cross-family per-query comparison
        flat_maps = attention.view(b, f * q, n)
        flat_norm = F.normalize(flat_maps, p=2, dim=-1, eps=1.0e-8)
        query_cos = torch.einsum("bid,bjd->bij", flat_norm, flat_norm)
        # Mask within-family comparisons (i, j with same family index): we want cross-family only.
        family_idx = torch.arange(f, device=attention.device).repeat_interleave(q)
        cross_family_mask = (family_idx.view(-1, 1) != family_idx.view(1, -1)).to(attention.dtype)
        cross_family_mask = cross_family_mask.unsqueeze(0)
        query_distance = (1.0 - query_cos) * cross_family_mask
        query_distance_max = query_distance.amax(dim=(1, 2))

        # Covariance scalar of attended-mean coordinates: trace of cov over D, divided by D
        cov_trace = attended_residual.square().sum(dim=2).mean(dim=1)  # (B,)

        diag_features = torch.cat(
            [
                attended_mean,
                attended_std,
                js_pairs if js_pairs.shape[-1] > 0 else attention.new_zeros(b, self._family_pair_count),
                entropy_per_family,
                entropy_variance.unsqueeze(-1),
                cos_distance_max.unsqueeze(-1),
                query_distance_max.unsqueeze(-1),
                cov_trace.unsqueeze(-1),
            ],
            dim=-1,
        )
        scalar_outputs = {
            "attention_js_divergence_mean": js_mean,
            "attention_js_divergence_max": js_max,
            "attention_entropy_variance": entropy_variance,
            "attention_entropy_mean": entropy.mean(dim=(1, 2)),
            "family_map_cosine_distance_mean": cos_distance_mean,
            "family_map_cosine_distance_max": cos_distance_max,
            "query_map_cosine_distance_max": query_distance_max,
            "attended_residual_norm": attended_residual.norm(dim=-1).mean(dim=1),
            "attended_covariance_trace": cov_trace,
            "attended_mean_norm": attended_mean.norm(dim=-1),
        }
        return diag_features, scalar_outputs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        tokens = self.tokenizer(x)
        attention, attended = self._attention(tokens)
        features, scalars = self._disagreement(attention, attended)
        logits = self.classifier(features).view(-1)
        return {
            "logits": logits,
            "attention": attention,
            "attended_values": attended,
            "disagreement_features": features,
            **scalars,
        }


def build_attention_disagreement_residual_network_from_config(
    config: dict[str, Any],
) -> AttentionDisagreementResidualNetwork:
    cfg = dict(config)
    token_dim = int(cfg.get("token_dim", cfg.get("channels", 64)))
    return AttentionDisagreementResidualNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=token_dim,
        family_count=int(cfg.get("family_count", 4)),
        query_count=int(cfg.get("query_count", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        head_hidden=int(cfg.get("head_hidden", cfg.get("hidden_dim", 128))),
        dropout=float(cfg.get("dropout", 0.1)),
        attention_dropout=float(cfg.get("attention_dropout", 0.0)),
    )
