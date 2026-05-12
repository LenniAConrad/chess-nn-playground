"""Credal Temperature Field Network for idea i220.

A shared board encoder produces a binary puzzle logit. A separate calibration head
predicts a sample-wise positive temperature T(x) (and a bounded smoothing factor
alpha(x)) that scale the raw logit. The model returns the calibrated logit and
exports the raw logit, T(x), and alpha(x) as diagnostics.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


class _Trunk(nn.Module):
    def __init__(self, input_channels: int, channels: int, depth: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(max(1, depth)):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class CredalTemperatureFieldNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        max_alpha: float = 0.4,
        temperature_floor: float = 0.25,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("CredalTemperatureFieldNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.max_alpha = float(max_alpha)
        self.temperature_floor = float(temperature_floor)
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)
        pooled_dim = channels * 2
        self.shared_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.logit_head = nn.Linear(hidden_dim, 1)
        self.temperature_head = nn.Linear(hidden_dim, 1)
        self.smoothing_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)
        shared = self.shared_head(pooled)
        raw_logit = self.logit_head(shared).view(-1)
        temperature = F.softplus(self.temperature_head(shared).view(-1)) + self.temperature_floor
        smoothing = self.max_alpha * torch.sigmoid(self.smoothing_head(shared).view(-1))
        calibrated_logit = raw_logit / temperature
        scaled_logit = (1.0 - smoothing) * calibrated_logit
        prob = torch.sigmoid(scaled_logit).clamp(1.0e-6, 1.0 - 1.0e-6)
        entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
        return {
            "logits": scaled_logit,
            "raw_logits": raw_logit,
            "calibrated_logits": calibrated_logit,
            "credal_temperature": temperature,
            "credal_smoothing": smoothing,
            "credal_entropy": entropy,
            "credal_temperature_log": temperature.log(),
        }


def build_credal_temperature_field_network_from_config(config: dict[str, Any]) -> CredalTemperatureFieldNetwork:
    cfg = dict(config)
    return CredalTemperatureFieldNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        max_alpha=float(cfg.get("max_alpha", 0.4)),
        temperature_floor=float(cfg.get("temperature_floor", 0.25)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
