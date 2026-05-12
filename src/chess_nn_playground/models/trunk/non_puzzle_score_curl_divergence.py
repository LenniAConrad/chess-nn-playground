"""Non-Puzzle Score Curl-Divergence Bottleneck for idea i216.

Implements the ``score field curl-divergence bottleneck'' thesis: the
network predicts a 2D vector score field on the 8x8 board, derives its
discrete divergence and curl, and channels only the curl/divergence
summary statistics through the puzzle classifier head. The architecture
is materially distinct from the shared research-packet probe.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models._packet_bespoke_base import (
    BoardConvStem,
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


def _discrete_partials(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pad = F.pad(field, (1, 1, 1, 1), mode="replicate")
    dx = 0.5 * (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2])
    dy = 0.5 * (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1])
    return dx, dy


class _ScoreField(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, 2, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.body(feats)


class NonPuzzleScoreCurlDivergenceBottleneck(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("NonPuzzleScoreCurlDivergenceBottleneck supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.field = _ScoreField(channels, dropout)
        bottleneck_dim = 12
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)
        field = self.field(feats)
        u = field[:, 0]
        v = field[:, 1]
        du_dx, du_dy = _discrete_partials(u)
        dv_dx, dv_dy = _discrete_partials(v)
        divergence = du_dx + dv_dy
        curl = dv_dx - du_dy
        bottleneck = torch.stack(
            [
                divergence.mean(dim=(1, 2)),
                divergence.abs().mean(dim=(1, 2)),
                divergence.amax(dim=(1, 2)),
                divergence.amin(dim=(1, 2)),
                curl.mean(dim=(1, 2)),
                curl.abs().mean(dim=(1, 2)),
                curl.amax(dim=(1, 2)),
                curl.amin(dim=(1, 2)),
                divergence.flatten(1).std(dim=1),
                curl.flatten(1).std(dim=1),
                field.norm(dim=1).mean(dim=(1, 2)),
                (curl.abs() + divergence.abs()).mean(dim=(1, 2)),
            ],
            dim=1,
        )
        bottleneck = self.bottleneck_norm(bottleneck)
        logits = format_logits(self.head(bottleneck), self.num_classes)
        return {
            "logits": logits,
            "score_field_norm": field.norm(dim=1).mean(dim=(1, 2)),
            "score_divergence_mean": divergence.mean(dim=(1, 2)),
            "score_divergence_abs_mean": divergence.abs().mean(dim=(1, 2)),
            "score_curl_mean": curl.mean(dim=(1, 2)),
            "score_curl_abs_mean": curl.abs().mean(dim=(1, 2)),
            "curl_divergence_total": (curl.abs() + divergence.abs()).mean(dim=(1, 2)),
            "score_divergence_field": divergence,
            "score_curl_field": curl,
        }


def build_non_puzzle_score_curl_divergence_bottleneck_from_config(config: dict[str, Any]) -> NonPuzzleScoreCurlDivergenceBottleneck:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return NonPuzzleScoreCurlDivergenceBottleneck(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
