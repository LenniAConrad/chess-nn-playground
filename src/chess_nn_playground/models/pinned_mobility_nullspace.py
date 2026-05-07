"""Pinned Mobility Nullspace Network for idea i208.

Implements the ``apparent defenders whose mobility lies in a nullspace``
thesis: the network estimates per-piece soft mobility scores and a soft
pin mask whose mobility-direction is treated as a nullspace; pieces in
that nullspace are scored as effectively immobile defenders. The
architecture is materially distinct from the shared research-packet
probe.
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
    us_them_piece_planes,
)


class PinnedNullspaceProbe(nn.Module):
    def __init__(self, channels: int, mobility_rank: int, dropout: float) -> None:
        super().__init__()
        self.mobility_rank = int(mobility_rank)
        self.mobility_proj = nn.Conv2d(channels, mobility_rank, kernel_size=3, padding=1)
        self.pin_proj = nn.Conv2d(channels, mobility_rank, kernel_size=3, padding=1)
        self.role_proj = nn.Conv2d(channels, 6, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, feats: torch.Tensor, occupancy: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.dropout(feats)
        mobility = torch.tanh(self.mobility_proj(feats))
        pin_axis = self.pin_proj(feats)
        pin_axis = pin_axis / pin_axis.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
        projection = (mobility * pin_axis).sum(dim=1, keepdim=True) * pin_axis
        nullspace = mobility - projection
        nullspace_energy = nullspace.square().sum(dim=1) * occupancy
        free_mobility_energy = mobility.square().sum(dim=1) * occupancy
        pin_alignment = (mobility * pin_axis).sum(dim=1).abs() * occupancy
        roles = F.softmax(self.role_proj(feats), dim=1)
        return {
            "mobility": mobility,
            "nullspace": nullspace,
            "pin_axis": pin_axis,
            "nullspace_energy": nullspace_energy,
            "free_mobility_energy": free_mobility_energy,
            "pin_alignment_field": pin_alignment,
            "role_distribution": roles,
        }


class PinnedMobilityNullspaceNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        mobility_rank: int = 8,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("PinnedMobilityNullspaceNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.probe = PinnedNullspaceProbe(channels, mobility_rank, dropout)
        head_in = 6 + 8 + channels
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)
        us, them = us_them_piece_planes(x, self.input_channels)
        occupancy = (us.sum(dim=1) + them.sum(dim=1)).clamp(0.0, 1.0)
        probe = self.probe(feats, occupancy)
        per_role_pinned = (probe["role_distribution"] * probe["nullspace_energy"].unsqueeze(1)).sum(dim=(2, 3))
        nullspace_total = probe["nullspace_energy"].sum(dim=(1, 2))
        free_mobility_total = probe["free_mobility_energy"].sum(dim=(1, 2))
        pin_alignment_total = probe["pin_alignment_field"].sum(dim=(1, 2))
        nullspace_ratio = nullspace_total / (free_mobility_total + nullspace_total + 1.0e-6)
        nullspace_top = probe["nullspace_energy"].flatten(1).topk(8, dim=1).values
        feats_pool = feats.mean(dim=(2, 3))
        readout = torch.cat([per_role_pinned, nullspace_top, feats_pool], dim=1)
        readout = self.head_norm(readout)
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "pin_nullspace_energy": nullspace_total,
            "free_mobility_energy_total": free_mobility_total,
            "pin_alignment_total": pin_alignment_total,
            "pin_nullspace_ratio": nullspace_ratio,
            "per_role_pin_pressure": per_role_pinned,
            "topk_nullspace_pressure": nullspace_top,
        }


def build_pinned_mobility_nullspace_network_from_config(config: dict[str, Any]) -> PinnedMobilityNullspaceNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return PinnedMobilityNullspaceNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        mobility_rank=int(cfg.get("mobility_rank", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
