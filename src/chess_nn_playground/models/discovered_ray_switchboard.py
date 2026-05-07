"""Discovered-Ray Switchboard Network for idea i206.

Implements the ``critical line appears only after a blocker moves``
thesis: candidate blocker squares are detected on the eight ray
directions, and removing each blocker is treated as a switchboard event
that may expose new tactical pressure. The architecture is materially
distinct from the shared research-packet probe.
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


def _direction_kernels() -> torch.Tensor:
    kernels = []
    for dr, df in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        k = torch.zeros(3, 3)
        k[1 + dr, 1 + df] = 1.0
        kernels.append(k)
    return torch.stack(kernels, dim=0).unsqueeze(1)


class RaySwitchboard(nn.Module):
    def __init__(self, channels: int, num_directions: int = 8, ray_steps: int = 4) -> None:
        super().__init__()
        self.num_directions = int(num_directions)
        self.ray_steps = int(ray_steps)
        self.register_buffer("direction_kernels", _direction_kernels(), persistent=False)
        self.blocker_proj = nn.Conv2d(channels, num_directions, kernel_size=1)
        self.exposure_proj = nn.Conv2d(channels, num_directions, kernel_size=1)
        self.gate = nn.Conv2d(num_directions * 2, num_directions, kernel_size=1)

    def _ray_propagate(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = x.shape
        x_flat = x.view(batch * channels, 1, h, w)
        propagated = []
        for direction in range(self.num_directions):
            kernel = self.direction_kernels[direction : direction + 1]
            cur = x_flat
            cumulative = torch.zeros_like(cur)
            for _ in range(self.ray_steps):
                cur = F.conv2d(cur, kernel, padding=1)
                cumulative = cumulative + cur
            propagated.append(cumulative.view(batch, channels, h, w))
        return torch.stack(propagated, dim=2)

    def forward(self, feats: torch.Tensor, occupancy: torch.Tensor) -> dict[str, torch.Tensor]:
        blocker_score = torch.sigmoid(self.blocker_proj(feats)) * occupancy.unsqueeze(1)
        exposure_score = torch.sigmoid(self.exposure_proj(feats))
        propagated_exposure = self._ray_propagate(exposure_score)
        propagated_exposure = propagated_exposure.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
        switchboard = blocker_score * propagated_exposure
        gate = torch.sigmoid(self.gate(torch.cat([blocker_score, propagated_exposure], dim=1)))
        gated = switchboard * gate
        return {
            "blocker_score": blocker_score,
            "exposure_score": exposure_score,
            "switchboard_activation": gated,
        }


class DiscoveredRaySwitchboardNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        ray_steps: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("DiscoveredRaySwitchboardNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.switchboard = RaySwitchboard(channels=channels, num_directions=8, ray_steps=ray_steps)
        head_in = 8 * 4 + channels + 4
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
        sb = self.switchboard(feats, occupancy)
        switchboard_act = sb["switchboard_activation"]
        per_dir_mean = switchboard_act.mean(dim=(2, 3))
        per_dir_max = switchboard_act.amax(dim=(2, 3))
        per_dir_blocker_mean = sb["blocker_score"].mean(dim=(2, 3))
        per_dir_exposure_mean = sb["exposure_score"].mean(dim=(2, 3))
        feats_pool = feats.mean(dim=(2, 3))
        switchboard_total = switchboard_act.sum(dim=(1, 2, 3))
        switchboard_entropy_field = switchboard_act.sum(dim=1).flatten(1)
        switchboard_entropy_field = F.softmax(switchboard_entropy_field, dim=-1)
        switchboard_entropy = -(switchboard_entropy_field.clamp_min(1.0e-6).log() * switchboard_entropy_field).sum(dim=-1)
        ray_imbalance = (per_dir_mean.amax(dim=1) - per_dir_mean.amin(dim=1))
        readout = torch.cat(
            [
                per_dir_mean,
                per_dir_max,
                per_dir_blocker_mean,
                per_dir_exposure_mean,
                feats_pool,
                switchboard_total.unsqueeze(-1),
                switchboard_entropy.unsqueeze(-1),
                ray_imbalance.unsqueeze(-1),
                occupancy.mean(dim=(1, 2)).unsqueeze(-1),
            ],
            dim=1,
        )
        readout = self.head_norm(readout)
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "switchboard_activation_per_direction": per_dir_mean,
            "switchboard_entropy": switchboard_entropy,
            "discovered_ray_energy": switchboard_total,
            "blocker_density_per_direction": per_dir_blocker_mean,
            "exposure_density_per_direction": per_dir_exposure_mean,
            "ray_directional_imbalance": ray_imbalance,
        }


def build_discovered_ray_switchboard_network_from_config(config: dict[str, Any]) -> DiscoveredRaySwitchboardNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return DiscoveredRaySwitchboardNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        ray_steps=int(cfg.get("ray_steps", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
