"""Role-Counterfactual Necessity Network for idea i211.

Implements the ``role-counterfactual necessity'' thesis: the network
computes a base puzzle score, then re-evaluates the position six times
with each piece role (pawn, knight, bishop, rook, queen, king) softly
masked. The drop in puzzle evidence under each role-removal becomes a
necessity score, and the final logit is conditioned on the necessity
profile. The architecture is materially distinct from the shared
research-packet probe.
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


ROLE_NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")


class _CounterfactualHead(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pool_norm = nn.LayerNorm(channels * 2)
        self.head = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([feats.mean(dim=(2, 3)), feats.amax(dim=(2, 3))], dim=-1)
        pooled = self.pool_norm(pooled)
        return self.head(pooled).squeeze(-1)


class RoleCounterfactualNecessityNetwork(nn.Module):
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
            raise ValueError("RoleCounterfactualNecessityNetwork supports the puzzle_binary one-logit contract")
        if input_channels < 12:
            raise ValueError("RoleCounterfactualNecessityNetwork requires the simple_18 piece planes (>=12 channels)")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.head = _CounterfactualHead(channels, hidden_dim, dropout)
        readout = len(ROLE_NAMES) + 4
        self.necessity_norm = nn.LayerNorm(readout)
        self.necessity_head = nn.Sequential(
            nn.Linear(readout, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _forward_logit(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stem(x)
        return self.head(feats)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        base_logit = self._forward_logit(x)
        necessities = []
        masked_logits = []
        for role_idx, _ in enumerate(ROLE_NAMES):
            mask = torch.ones(self.input_channels, device=x.device, dtype=x.dtype)
            mask[role_idx] = 0.0
            mask[role_idx + 6] = 0.0
            masked_x = x * mask.view(1, -1, 1, 1)
            logit_masked = self._forward_logit(masked_x)
            masked_logits.append(logit_masked)
            necessities.append(base_logit - logit_masked)
        necessity = torch.stack(necessities, dim=1)
        masked = torch.stack(masked_logits, dim=1)
        max_necessity = necessity.amax(dim=1)
        min_necessity = necessity.amin(dim=1)
        necessity_entropy = F.softmax(necessity, dim=1)
        necessity_entropy = -(necessity_entropy.clamp_min(1.0e-6).log() * necessity_entropy).sum(dim=1)
        readout = torch.cat(
            [
                necessity,
                base_logit.unsqueeze(-1),
                max_necessity.unsqueeze(-1),
                min_necessity.unsqueeze(-1),
                necessity_entropy.unsqueeze(-1),
            ],
            dim=1,
        )
        readout = self.necessity_norm(readout)
        logits = format_logits(self.necessity_head(readout), self.num_classes)
        return {
            "logits": logits,
            "base_logit": base_logit,
            "role_masked_logits": masked,
            "role_necessity": necessity,
            "max_role_necessity": max_necessity,
            "min_role_necessity": min_necessity,
            "role_necessity_entropy": necessity_entropy,
        }


def build_role_counterfactual_necessity_network_from_config(config: dict[str, Any]) -> RoleCounterfactualNecessityNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return RoleCounterfactualNecessityNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
