from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor

from chess_nn_playground.models.trunk._gpt_research_blocks import (
    CompactBoardEncoder,
    bernoulli_kl_from_logits,
    binary_concrete_gate,
)


class ConditionalSurprisalGatePuzzleNet(nn.Module):
    """Gate-only classifier conditioned on a weak board-statistics prior."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 96,
        depth: int = 3,
        hidden_dim: int = 128,
        gate_dim: int = 64,
        tau: float = 0.8,
        hard_gate: bool = True,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("ConditionalSurprisalGatePuzzleNet supports puzzle_binary single-logit output")
        self.encoder = CompactBoardEncoder(input_channels, channels, depth, hidden_dim, dropout, use_batchnorm)
        self.prior = nn.Sequential(
            nn.Linear(17, max(16, gate_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, gate_dim // 2), 1),
        )
        self.posterior = nn.Linear(hidden_dim, gate_dim)
        self.prior_to_gate = nn.Linear(1, gate_dim)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(gate_dim, 1),
        )
        self.posterior_probe = nn.Linear(hidden_dim, 1)
        self.tau = float(tau)
        self.hard_gate = bool(hard_gate)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _board, hidden, stats = self.encoder(x)
        prior_logits = self.prior(stats).view(-1)
        posterior_probe = self.posterior_probe(hidden).view(-1)
        surprisal = bernoulli_kl_from_logits(posterior_probe, prior_logits)
        gate_logits = self.posterior(hidden) - self.prior_to_gate(prior_logits.unsqueeze(1))
        gate = binary_concrete_gate(gate_logits, tau=self.tau, hard=self.hard_gate, training=self.training)
        logits = self.gate_head(gate).view(-1)
        return {
            "logits": logits,
            "prior_logits": prior_logits,
            "posterior_logits": posterior_probe,
            "gate_logit_mean": gate_logits.mean(dim=1),
            "gate_mean": gate.mean(dim=1),
            "conditional_surprisal": surprisal,
        }


def build_conditional_surprisal_gate_from_config(config: dict[str, Any]) -> ConditionalSurprisalGatePuzzleNet:
    return ConditionalSurprisalGatePuzzleNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 96)),
        depth=int(config.get("depth", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        gate_dim=int(config.get("gate_dim", 64)),
        tau=float(config.get("tau", 0.8)),
        hard_gate=bool(config.get("hard_gate", True)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
