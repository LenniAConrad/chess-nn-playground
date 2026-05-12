"""Shared building blocks for bespoke research-packet idea architectures.

These helpers are intentionally compact: each idea-specific model file
imports them and adds its own distinctive mechanism on top. The trunk
mirrors the ``compact convolutional square encoder`` described in the
batch-4 research packet markdown, so every architecture downstream can
focus on its idea-specific differentiator.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardConvStem,
    BoardTensorSpec,
    require_board_tensor,
)


__all__ = [
    "BoardTensorSpec",
    "BoardConvStem",
    "require_board_tensor",
    "format_logits",
    "side_to_move_field",
    "us_them_piece_planes",
    "rank_file_grid",
    "soft_topk_pool",
]


def format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def side_to_move_field(x: torch.Tensor, input_channels: int) -> torch.Tensor:
    """Return a (B, 1, 8, 8) side-to-move field aligned with the trunk output."""
    if input_channels >= 13:
        return x[:, 12:13].clamp(0.0, 1.0)
    return x.new_ones(x.shape[0], 1, 8, 8)


def us_them_piece_planes(x: torch.Tensor, input_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (us, them) piece planes shape (B, 6, 8, 8) for simple_18 boards."""
    if input_channels >= 12:
        white = x[:, 0:6].clamp(0.0, 1.0)
        black = x[:, 6:12].clamp(0.0, 1.0)
    else:
        white = x.new_zeros(x.shape[0], 6, 8, 8)
        black = x.new_zeros(x.shape[0], 6, 8, 8)
    side = side_to_move_field(x, input_channels)
    us = side * white + (1.0 - side) * black
    them = side * black + (1.0 - side) * white
    return us, them


def rank_file_grid(batch: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    rank = torch.arange(8, device=device, dtype=dtype).view(1, 1, 8, 1).expand(batch, 1, 8, 8) / 7.0
    file = torch.arange(8, device=device, dtype=dtype).view(1, 1, 1, 8).expand(batch, 1, 8, 8) / 7.0
    return rank, file


def soft_topk_pool(features: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    """Score-weighted top-k pool over flattened (B, N, D) tensor."""
    if features.shape[1] == 0:
        return features.new_zeros(features.shape[0], features.shape[-1])
    k = max(1, min(int(k), features.shape[1]))
    top = scores.topk(k, dim=1)
    weights = F.softmax(top.values, dim=1).unsqueeze(-1)
    gathered = torch.gather(features, 1, top.indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
    return (gathered * weights).sum(dim=1)
