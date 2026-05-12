"""Stable-Rank Multiscale Network (idea i238).

Computes the stable rank ||A||_F^2 / ||A||_2^2 of a learned chess interaction
matrix at multiple block-scales: full 64x64, 4 quadrant-blocks (32x32 each),
16 sub-blocks (16x16 each). Stable rank is a continuous, differentiable
relaxation of rank that satisfies stable_rank(A) <= rank(A); collapsing stable
rank under a chess-meaningful re-blocking is a different inductive bias from
spectrum, condition number, or nuclear norm.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem


def _stable_rank(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable rank ||M||_F^2 / ||M||_2^2 along the last two dims."""
    fro_sq = (M ** 2).sum(dim=(-2, -1))
    # ||M||_2 via top singular value -- cheap power iteration is differentiable; we use linalg.svdvals here.
    sigma = torch.linalg.svdvals(M)[..., 0]
    return fro_sq / (sigma ** 2 + eps)


class StableRankMultiscaleNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 32,
        stem_depth: int = 2,
        hidden_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = BoardConvStem(
            input_channels=input_channels, channels=channels, depth=stem_depth
        )
        # Build a 64x64 interaction by outer product of two 64-d vectors per channel,
        # then sum over channels (gives a learned bilinear interaction).
        self.left_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.right_proj = nn.Conv2d(channels, channels, kernel_size=1)
        # Multiscale stable-rank features: 1 (64x64) + 4 (32x32 blocks) + 16 (16x16 blocks).
        feature_dim = 1 + 4 + 16 + channels  # plus pooled channel features
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def build_interaction(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, 8, 8). Build bilinear (B, 64, 64).
        B, C, H, W = feat.shape
        L = self.left_proj(feat).view(B, C, H * W)         # (B, C, 64)
        R = self.right_proj(feat).view(B, C, H * W)        # (B, C, 64)
        # Sum over channels: M[i,j] = sum_c L[c,i] * R[c,j].
        M = torch.einsum("bcs,bct->bst", L, R)             # (B, 64, 64)
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        M = self.build_interaction(feat)                   # (B, 64, 64)
        # Full-scale stable rank.
        sr_full = _stable_rank(M).unsqueeze(-1)            # (B, 1)
        # 4 quadrant blocks.
        Q = M.view(M.shape[0], 2, 32, 2, 32).permute(0, 1, 3, 2, 4).reshape(M.shape[0], 4, 32, 32)
        sr_quad = _stable_rank(Q)                           # (B, 4)
        # 16 sub-blocks.
        S = M.view(M.shape[0], 4, 16, 4, 16).permute(0, 1, 3, 2, 4).reshape(M.shape[0], 16, 16, 16)
        sr_sub = _stable_rank(S)                            # (B, 16)
        pooled = feat.mean(dim=(-2, -1))                   # (B, C)
        feature = torch.cat([sr_full, sr_quad, sr_sub, pooled], dim=-1)
        out = self.head(feature)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


def build_stable_rank_multiscale_from_config(config: dict[str, Any]) -> StableRankMultiscaleNetwork:
    return StableRankMultiscaleNetwork(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 32)),
        stem_depth=int(config.get("stem_depth", 2)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        dropout=float(config.get("dropout", 0.1)),
    )
