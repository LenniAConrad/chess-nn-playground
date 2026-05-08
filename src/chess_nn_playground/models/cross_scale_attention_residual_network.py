"""Cross-Scale Attention Residual Network for idea i104.

Builds 64 fine square tokens and K coarse tokens (averaged over fixed
non-overlapping patches of the 8x8 board), computes the actual fine->fine
attention map and a coarse-anchored rank-K prediction of that map factored
through the coarse pivots, and classifies puzzle-likeness from the residual
attention map A_actual - A_predicted.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


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
        self.register_buffer("coords", _board_coords(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_values = x.flatten(2).transpose(1, 2)
        coords = self.coords.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.mlp(torch.cat([square_values, coords], dim=-1))


def _coarse_indices(coarse_scale: int) -> torch.Tensor:
    if coarse_scale not in (2, 4):
        raise ValueError("coarse_scale must be 2 or 4")
    side = 8 // coarse_scale
    num_patches = side * side
    cell_count = coarse_scale * coarse_scale
    indices = torch.zeros(num_patches, cell_count, dtype=torch.long)
    for p in range(num_patches):
        pr, pf = divmod(p, side)
        cell = 0
        for dr in range(coarse_scale):
            for df in range(coarse_scale):
                r = pr * coarse_scale + dr
                f = pf * coarse_scale + df
                indices[p, cell] = r * 8 + f
                cell += 1
    return indices


class CrossScaleAttentionResidualNetwork(nn.Module):
    """Residual between actual fine->fine attention and a coarse-anchored prediction."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        token_dim: int = 64,
        coarse_scale: int = 2,
        hidden_dim: int = 96,
        head_hidden: int = 128,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        coarse_residual_channels: int = 32,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "CrossScaleAttentionResidualNetwork supports the puzzle_binary one-logit contract"
            )
        if token_dim < 1:
            raise ValueError("token_dim must be positive")
        self.spec = BoardTensorSpec(input_channels=int(input_channels))
        self.token_dim = int(token_dim)
        self.coarse_scale = int(coarse_scale)

        self.tokenizer = _SquareTokenizer(
            input_channels=int(input_channels),
            token_dim=self.token_dim,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )

        coarse_indices = _coarse_indices(self.coarse_scale)
        self.register_buffer("coarse_indices", coarse_indices, persistent=False)
        self.num_coarse = int(coarse_indices.shape[0])

        self.fine_query = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.fine_key = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.coarse_query = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.coarse_key = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.coarse_norm = nn.LayerNorm(self.token_dim)
        self.attention_dropout = (
            nn.Dropout(float(attention_dropout)) if attention_dropout > 0 else nn.Identity()
        )

        # Treat the residual map as 64 source-square channels over an 8x8 target board.
        self.residual_conv = nn.Sequential(
            nn.Conv2d(64, int(coarse_residual_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(coarse_residual_channels)),
            nn.GELU(),
            nn.Conv2d(
                int(coarse_residual_channels),
                int(coarse_residual_channels),
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(coarse_residual_channels)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        scalar_feature_dim = 8
        diag_dim = int(coarse_residual_channels) + scalar_feature_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(diag_dim),
            nn.Linear(diag_dim, int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(head_hidden), 1),
        )

    def _coarse_tokens(self, fine_tokens: torch.Tensor) -> torch.Tensor:
        b, _, d = fine_tokens.shape
        idx = self.coarse_indices.to(device=fine_tokens.device)
        gathered = fine_tokens[:, idx.view(-1), :].view(b, idx.shape[0], idx.shape[1], d)
        return self.coarse_norm(gathered.mean(dim=2))

    def _residual_features(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b = residual.shape[0]
        abs_r = residual.abs()
        per_row_l1 = abs_r.sum(dim=-1)
        total_energy = per_row_l1.sum(dim=-1)
        max_abs = abs_r.amax(dim=(-1, -2))
        diag = residual.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        eye = torch.eye(64, device=residual.device, dtype=residual.dtype).unsqueeze(0)
        off_diag_energy = (abs_r * (1.0 - eye)).sum(dim=(-1, -2))
        asym = (residual - residual.transpose(-1, -2)).abs().mean(dim=(-1, -2))
        frobenius = residual.square().sum(dim=(-1, -2)).clamp_min(0.0).sqrt()
        row_norm = abs_r / abs_r.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        row_entropy = -(row_norm * row_norm.clamp_min(1.0e-8).log()).sum(dim=-1) / math.log(64.0)
        row_entropy_mean = row_entropy.mean(dim=-1)
        row_entropy_var = row_entropy.var(dim=-1, unbiased=False)
        scalar_features = torch.stack(
            [
                total_energy,
                off_diag_energy,
                max_abs,
                diag,
                asym,
                frobenius,
                row_entropy_mean,
                row_entropy_var,
            ],
            dim=-1,
        )
        residual_image = residual.view(b, 64, 8, 8)
        scalar_outputs = {
            "residual_total_energy": total_energy,
            "residual_off_diagonal_energy": off_diag_energy,
            "residual_max_abs": max_abs,
            "residual_self_diagonal_mean": diag,
            "residual_asymmetry": asym,
            "residual_frobenius": frobenius,
            "residual_row_entropy_mean": row_entropy_mean,
            "residual_row_entropy_variance": row_entropy_var,
            "residual_per_source_l1": per_row_l1,
        }
        return scalar_features, residual_image, scalar_outputs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        fine_tokens = self.tokenizer(x)
        coarse_tokens = self._coarse_tokens(fine_tokens)

        scale = 1.0 / math.sqrt(float(self.token_dim))
        q_fine = self.fine_query(fine_tokens)
        k_fine = self.fine_key(fine_tokens)
        q_coarse = self.coarse_query(coarse_tokens)
        k_coarse = self.coarse_key(coarse_tokens)

        scores_actual = torch.matmul(q_fine, k_fine.transpose(-1, -2)) * scale
        attn_actual = F.softmax(scores_actual, dim=-1)
        attn_actual = self.attention_dropout(attn_actual)

        scores_fine_to_coarse = torch.matmul(q_fine, k_coarse.transpose(-1, -2)) * scale
        a_fine_to_coarse = F.softmax(scores_fine_to_coarse, dim=-1)
        scores_coarse_to_fine = torch.matmul(q_coarse, k_fine.transpose(-1, -2)) * scale
        a_coarse_to_fine = F.softmax(scores_coarse_to_fine, dim=-1)
        attn_predicted = torch.matmul(a_fine_to_coarse, a_coarse_to_fine)

        residual = attn_actual - attn_predicted

        scalar_features, residual_image, scalar_outputs = self._residual_features(residual)
        conv_features = self.residual_conv(residual_image)
        features = torch.cat([conv_features, scalar_features], dim=-1)
        logits = self.classifier(features).view(-1)

        return {
            "logits": logits,
            "attention_actual": attn_actual,
            "attention_predicted": attn_predicted,
            "residual_attention": residual,
            "fine_to_coarse_attention": a_fine_to_coarse,
            "coarse_to_fine_attention": a_coarse_to_fine,
            "residual_features": features,
            **scalar_outputs,
        }


def build_cross_scale_attention_residual_network_from_config(
    config: dict[str, Any],
) -> CrossScaleAttentionResidualNetwork:
    cfg = dict(config)
    token_dim = int(cfg.get("token_dim", cfg.get("channels", 64)))
    return CrossScaleAttentionResidualNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        token_dim=token_dim,
        coarse_scale=int(cfg.get("coarse_scale", 2)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        head_hidden=int(cfg.get("head_hidden", cfg.get("hidden_dim", 128))),
        dropout=float(cfg.get("dropout", 0.1)),
        attention_dropout=float(cfg.get("attention_dropout", 0.0)),
        coarse_residual_channels=int(cfg.get("coarse_residual_channels", 32)),
    )
