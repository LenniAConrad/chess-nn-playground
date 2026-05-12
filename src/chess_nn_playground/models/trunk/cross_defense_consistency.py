"""Cross-Defense Consistency Network for idea i204.

Implements the ``puzzle survives multiple independent defensive
interpretations`` thesis: K parallel defensive interpreter heads each
score the same board through different inductive biases (kernel size,
pooling, depth). Their agreement is what discriminates puzzles from
near-puzzles. The architecture is materially distinct from the shared
research-packet probe.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models._packet_bespoke_base import (
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


class DefensiveInterpreter(nn.Module):
    def __init__(self, input_channels: int, channels: int, kernel_size: int, hidden_dim: int, dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        norm_a = nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(min(8, channels), channels)
        norm_b = nn.BatchNorm2d(channels) if use_batchnorm else nn.GroupNorm(min(8, channels), channels)
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=padding),
            norm_a,
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            norm_b,
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        logit = self.head(feats).squeeze(-1)
        return logit, feats.mean(dim=(2, 3))


class CrossDefenseConsistencyNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_interpreters: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("CrossDefenseConsistencyNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        kernels = [1, 3, 3, 5][: max(1, num_interpreters)]
        if len(kernels) < num_interpreters:
            kernels = (kernels * ((num_interpreters // len(kernels)) + 1))[:num_interpreters]
        self.interpreters = nn.ModuleList(
            [
                DefensiveInterpreter(
                    input_channels=input_channels,
                    channels=max(8, channels // 2),
                    kernel_size=int(k),
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
                for k in kernels
            ]
        )
        self.consensus_norm = nn.LayerNorm(num_interpreters * 2 + 4)
        self.head = nn.Sequential(
            nn.Linear(num_interpreters * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        logits_list = []
        feats_list = []
        for interpreter in self.interpreters:
            logit, feats = interpreter(x)
            logits_list.append(logit)
            feats_list.append(feats.mean(dim=-1))
        per_interpreter = torch.stack(logits_list, dim=1)
        feature_summary = torch.stack(feats_list, dim=1)
        mean = per_interpreter.mean(dim=1)
        variance = per_interpreter.var(dim=1, unbiased=False)
        agreement = torch.exp(-variance)
        spread = per_interpreter.amax(dim=1) - per_interpreter.amin(dim=1)
        feat_var = feature_summary.var(dim=1, unbiased=False)
        readout = torch.cat(
            [
                per_interpreter,
                feature_summary,
                mean.unsqueeze(-1),
                variance.unsqueeze(-1),
                agreement.unsqueeze(-1),
                spread.unsqueeze(-1),
            ],
            dim=1,
        )
        readout = self.consensus_norm(readout)
        consensus_logit = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": consensus_logit,
            "per_interpreter_logits": per_interpreter,
            "cross_defense_mean": mean,
            "cross_defense_variance": variance,
            "cross_defense_agreement": agreement,
            "cross_defense_spread": spread,
            "feature_disagreement": feat_var,
        }


def build_cross_defense_consistency_network_from_config(config: dict[str, Any]) -> CrossDefenseConsistencyNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return CrossDefenseConsistencyNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        num_interpreters=int(cfg.get("num_interpreters", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
