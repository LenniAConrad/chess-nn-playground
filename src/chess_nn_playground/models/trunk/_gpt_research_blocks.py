from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _as_single_logit(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


def _side_canonical_piece_planes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    piece = x[:, : min(12, x.shape[1])].clamp(0.0, 1.0)
    if piece.shape[1] < 12:
        piece = F.pad(piece, (0, 0, 0, 0, 0, 12 - piece.shape[1]))
    first_side = piece[:, :6]
    second_side = piece[:, 6:12]
    if x.shape[1] <= 18 and x.shape[1] > 12:
        white_to_move = x[:, 12:13].mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)
        own = white_to_move * first_side + (1.0 - white_to_move) * second_side
        opp = white_to_move * second_side + (1.0 - white_to_move) * first_side
    else:
        own = first_side
        opp = second_side
    return piece, own, opp


class CompactBoardEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 96,
        depth: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.layers = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Linear(channels * 2 + 17, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.channels = channels
        self.hidden_dim = hidden_dim

    def board_stats(self, x: torch.Tensor) -> torch.Tensor:
        piece_planes, own_planes, opp_planes = _side_canonical_piece_planes(x)
        counts = piece_planes.flatten(2).sum(dim=2) / 16.0
        own_material = own_planes.flatten(2).sum(dim=(1, 2), keepdim=False).unsqueeze(1) / 16.0
        opp_material = opp_planes.flatten(2).sum(dim=(1, 2), keepdim=False).unsqueeze(1) / 16.0
        side = x[:, 12:13].mean(dim=(2, 3)) if x.shape[1] > 12 else counts.new_zeros(counts.shape[0], 1)
        material_delta = own_material - opp_material
        material_total = own_material + opp_material
        return torch.cat([counts, side, own_material, opp_material, material_delta, material_total], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.layers(x)
        pooled = torch.cat([board.mean(dim=(2, 3)), board.amax(dim=(2, 3))], dim=1)
        stats = self.board_stats(x)
        return board, self.projection(torch.cat([pooled, stats], dim=1)), stats


class DeterministicTacticalMaskBuilder(nn.Module):
    """Build current-board tactical support masks without search or source metadata."""

    def __init__(self) -> None:
        super().__init__()
        cross = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        diag = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        full = torch.ones(3, 3)
        kernels = torch.stack([cross, diag, full], dim=0).unsqueeze(1)
        self.register_buffer("kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        piece, own_planes, opp_planes = _side_canonical_piece_planes(x)
        own = own_planes.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        opp = opp_planes.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        occupancy = (own + opp).clamp(0.0, 1.0)
        empty = 1.0 - occupancy
        local = F.conv2d(occupancy, self.kernels.to(dtype=x.dtype), padding=1).clamp(0.0, 8.0) / 8.0
        own_pressure = F.conv2d(own, self.kernels[:1].to(dtype=x.dtype), padding=1).clamp(0.0, 4.0) / 4.0
        opp_pressure = F.conv2d(opp, self.kernels[1:2].to(dtype=x.dtype), padding=1).clamp(0.0, 4.0) / 4.0
        king_like = (piece[:, 5:6] + piece[:, 11:12]).clamp(0.0, 1.0)
        king_ring = F.max_pool2d(king_like, kernel_size=3, stride=1, padding=1)
        hanging = occupancy * F.relu(opp_pressure - own_pressure)
        mobility = empty * local[:, 2:3]
        return torch.cat(
            [
                occupancy,
                empty,
                local,
                own_pressure,
                opp_pressure,
                king_ring,
                hanging,
                mobility,
            ],
            dim=1,
        )


@dataclass(frozen=True)
class RobustBoardClassifierConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 96
    depth: int = 3
    hidden_dim: int = 128
    dropout: float = 0.1
    use_batchnorm: bool = True


def bernoulli_kl_from_logits(q_logits: torch.Tensor, p_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    p = torch.sigmoid(p_logits).clamp(eps, 1.0 - eps)
    return q * (q / p).log() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).log()


def binary_concrete_gate(logits: torch.Tensor, tau: float, hard: bool, training: bool) -> torch.Tensor:
    if training:
        u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
        noise = torch.log(u) - torch.log1p(-u)
        relaxed = torch.sigmoid((logits + noise) / max(tau, 1e-6))
    else:
        relaxed = torch.sigmoid(logits / max(tau, 1e-6))
    if not hard:
        return relaxed
    straight = (relaxed >= 0.5).to(relaxed.dtype)
    return straight.detach() - relaxed.detach() + relaxed
