"""Counterplay Insolvency Ledger model for idea i207.

Implements the ``defender's counterthreats remain solvent'' thesis:
the network measures attacker debits and defender credits as separate
streams and computes a soft solvency margin that becomes the puzzle
signal. The architecture is materially distinct from the shared
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
    us_them_piece_planes,
)


PIECE_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 4.0)


class LedgerStream(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, hidden_dim), hidden_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.value_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.norm(self.proj(feats)))
        h = self.dropout(h)
        per_square = self.value_head(h).squeeze(1)
        return per_square * mask, h.mean(dim=(2, 3))


class CounterplayInsolvencyLedger(nn.Module):
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
            raise ValueError("CounterplayInsolvencyLedger supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        ledger_dim = max(16, hidden_dim // 2)
        self.threat_stream = LedgerStream(channels, ledger_dim, dropout)
        self.counter_stream = LedgerStream(channels, ledger_dim, dropout)
        self.register_buffer("piece_values", torch.tensor(PIECE_VALUES), persistent=False)
        head_in = ledger_dim * 2 + 12
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
        us_mass = us.sum(dim=1).clamp(0.0, 1.0)
        them_mass = them.sum(dim=1).clamp(0.0, 1.0)
        threat_per_square, threat_summary = self.threat_stream(feats, them_mass)
        counter_per_square, counter_summary = self.counter_stream(feats, us_mass)
        threat_total = threat_per_square.sum(dim=(1, 2))
        counter_total = counter_per_square.sum(dim=(1, 2))
        ledger_balance = counter_total - threat_total
        insolvency = F.softplus(threat_total - counter_total)
        material_us = (us * self.piece_values.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
        material_them = (them * self.piece_values.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
        material_gap = material_us - material_them
        gap_field = (counter_per_square - threat_per_square)
        gap_max = gap_field.amin(dim=(1, 2))
        gap_var = gap_field.flatten(1).var(dim=1, unbiased=False)
        side_pressure_gap = (threat_total - counter_total) / (threat_total + counter_total + 1.0)
        readout = torch.cat(
            [
                threat_summary,
                counter_summary,
                threat_total.unsqueeze(-1),
                counter_total.unsqueeze(-1),
                ledger_balance.unsqueeze(-1),
                insolvency.unsqueeze(-1),
                material_us.unsqueeze(-1),
                material_them.unsqueeze(-1),
                material_gap.unsqueeze(-1),
                side_pressure_gap.unsqueeze(-1),
                gap_max.unsqueeze(-1),
                gap_var.unsqueeze(-1),
                us_mass.mean(dim=(1, 2)).unsqueeze(-1),
                them_mass.mean(dim=(1, 2)).unsqueeze(-1),
            ],
            dim=1,
        )
        readout = self.head_norm(readout)
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "threat_debit": threat_total,
            "counter_credit": counter_total,
            "ledger_balance": ledger_balance,
            "insolvency_score": insolvency,
            "side_pressure_gap": side_pressure_gap,
            "material_gap": material_gap,
            "ledger_gap_min": gap_max,
            "ledger_gap_variance": gap_var,
        }


def build_counterplay_insolvency_ledger_from_config(config: dict[str, Any]) -> CounterplayInsolvencyLedger:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return CounterplayInsolvencyLedger(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
