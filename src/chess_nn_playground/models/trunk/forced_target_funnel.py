"""Forced-Target Funnel Network for idea i213.

Implements the ``forced-target funnel'' thesis: tactical evidence is
expected to concentrate on a single target square in true puzzles. The
network produces a per-square evidence field, computes a soft argmax
target, and reads concentration / entropy statistics that quantify
funnel sharpness. The architecture is materially distinct from the
shared research-packet probe.
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


class _Funnel(nn.Module):
    def __init__(self, channels: int, evidence_channels: int, dropout: float) -> None:
        super().__init__()
        self.evidence = nn.Sequential(
            nn.Conv2d(channels, evidence_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, evidence_channels), evidence_channels),
            nn.GELU(),
            nn.Conv2d(evidence_channels, evidence_channels, kernel_size=1),
        )
        self.target_head = nn.Conv2d(evidence_channels, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = self.dropout(self.evidence(feats))
        target_logits = self.target_head(evidence).squeeze(1)
        flat = target_logits.flatten(1)
        target_probs = F.softmax(flat, dim=-1)
        evidence_strength = evidence.norm(dim=1).flatten(1)
        return {
            "evidence": evidence,
            "target_logits": target_logits,
            "target_probs": target_probs,
            "evidence_strength": evidence_strength,
        }


class ForcedTargetFunnelNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        evidence_channels: int = 32,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("ForcedTargetFunnelNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.funnel = _Funnel(channels, evidence_channels, dropout)
        head_in = evidence_channels + 6
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
        funnel = self.funnel(feats)
        target_probs = funnel["target_probs"]
        evidence_strength = funnel["evidence_strength"]
        evidence_flat = funnel["evidence"].flatten(2)
        target_pool = (evidence_flat * target_probs.unsqueeze(1)).sum(dim=-1)
        funnel_concentration = target_probs.amax(dim=-1)
        funnel_entropy = -(target_probs.clamp_min(1.0e-6).log() * target_probs).sum(dim=-1)
        evidence_total = evidence_strength.sum(dim=-1)
        evidence_at_target = (evidence_strength * target_probs).sum(dim=-1)
        target_index = target_probs.argmax(dim=-1).to(target_probs.dtype)
        readout = torch.cat(
            [
                target_pool,
                funnel_concentration.unsqueeze(-1),
                funnel_entropy.unsqueeze(-1),
                evidence_total.unsqueeze(-1),
                evidence_at_target.unsqueeze(-1),
                target_index.unsqueeze(-1),
                (evidence_at_target / (evidence_total + 1.0e-6)).unsqueeze(-1),
            ],
            dim=-1,
        )
        readout = self.head_norm(readout)
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "target_index": target_index,
            "funnel_concentration": funnel_concentration,
            "funnel_entropy": funnel_entropy,
            "evidence_at_target": evidence_at_target,
            "evidence_total": evidence_total,
            "evidence_concentration_ratio": evidence_at_target / (evidence_total + 1.0e-6),
            "target_probs": target_probs,
        }


def build_forced_target_funnel_network_from_config(config: dict[str, Any]) -> ForcedTargetFunnelNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return ForcedTargetFunnelNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        evidence_channels=int(cfg.get("evidence_channels", 32)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
