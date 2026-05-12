"""Defender Timing Schedule Network for idea i205.

Implements the ``defender resources cannot be scheduled before tactical
deadlines`` thesis: threats and defenders are extracted as token sets,
each threat has a soft tactical deadline, each defender has a soft
latency to reach that threat, and a Sinkhorn-normalised assignment
identifies overruns. The architecture is materially distinct from the
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


def _sinkhorn(log_alpha: torch.Tensor, iterations: int) -> torch.Tensor:
    log_alpha = log_alpha
    for _ in range(int(iterations)):
        log_alpha = log_alpha - log_alpha.logsumexp(dim=-1, keepdim=True)
        log_alpha = log_alpha - log_alpha.logsumexp(dim=-2, keepdim=True)
    return log_alpha.exp()


class TimingScheduler(nn.Module):
    def __init__(self, channels: int, num_threats: int, num_defenders: int, hidden_dim: int, dropout: float, sinkhorn_iters: int) -> None:
        super().__init__()
        self.num_threats = int(num_threats)
        self.num_defenders = int(num_defenders)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.threat_query = nn.Parameter(torch.randn(num_threats, channels) * 0.02)
        self.defender_query = nn.Parameter(torch.randn(num_defenders, channels) * 0.02)
        self.deadline_head = nn.Linear(channels, 1)
        self.latency_head = nn.Bilinear(channels, channels, 1)
        self.head_norm = nn.LayerNorm(num_threats + num_defenders + 6)
        self.head = nn.Sequential(
            nn.Linear(num_threats + num_defenders + 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _attend(self, tokens: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        attn_logits = torch.einsum("bsc,kc->bks", tokens, queries)
        attn = F.softmax(attn_logits, dim=-1)
        return torch.einsum("bks,bsc->bkc", attn, tokens)

    def forward(self, board_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        threats = self._attend(board_tokens, self.threat_query)
        defenders = self._attend(board_tokens, self.defender_query)
        deadlines = F.softplus(self.deadline_head(threats).squeeze(-1))
        threat_expanded = threats.unsqueeze(2).expand(-1, -1, self.num_defenders, -1).contiguous()
        defender_expanded = defenders.unsqueeze(1).expand(-1, self.num_threats, -1, -1).contiguous()
        latency = F.softplus(
            self.latency_head(
                threat_expanded.reshape(-1, threats.shape[-1]),
                defender_expanded.reshape(-1, defenders.shape[-1]),
            )
        ).view(-1, self.num_threats, self.num_defenders)
        log_alpha = -latency / (1.0 + latency.detach().mean(dim=(1, 2), keepdim=True))
        assignment = _sinkhorn(log_alpha, self.sinkhorn_iters)
        scheduled_latency = (assignment * latency).sum(dim=-1)
        overrun = (scheduled_latency - deadlines).clamp_min(0.0)
        coverage = assignment.sum(dim=-1)
        defender_load = assignment.sum(dim=-2)
        entropy = -(assignment.clamp_min(1.0e-6).log() * assignment).sum(dim=(1, 2))
        readout = torch.cat(
            [
                deadlines,
                defender_load,
                overrun.mean(dim=1, keepdim=True),
                overrun.amax(dim=1, keepdim=True),
                coverage.mean(dim=1, keepdim=True),
                scheduled_latency.mean(dim=1, keepdim=True),
                entropy.unsqueeze(-1),
                overrun.sum(dim=1, keepdim=True),
            ],
            dim=1,
        )
        readout = self.head_norm(readout)
        logit = self.head(readout).squeeze(-1)
        return {
            "logit": logit,
            "deadlines": deadlines,
            "scheduled_latency": scheduled_latency,
            "schedule_overrun": overrun,
            "schedule_overrun_total": overrun.sum(dim=1),
            "schedule_coverage": coverage,
            "defender_load": defender_load,
            "schedule_entropy": entropy,
        }


class DefenderTimingScheduleNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_threats: int = 6,
        num_defenders: int = 6,
        sinkhorn_iters: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("DefenderTimingScheduleNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.scheduler = TimingScheduler(
            channels=channels,
            num_threats=num_threats,
            num_defenders=num_defenders,
            hidden_dim=hidden_dim,
            dropout=dropout,
            sinkhorn_iters=sinkhorn_iters,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)
        tokens = feats.flatten(2).transpose(1, 2)
        out = self.scheduler(tokens)
        logits = format_logits(out["logit"].unsqueeze(-1), self.num_classes)
        return {
            "logits": logits,
            "deadlines": out["deadlines"],
            "scheduled_latency": out["scheduled_latency"],
            "schedule_overrun": out["schedule_overrun"],
            "schedule_overrun_total": out["schedule_overrun_total"],
            "schedule_coverage": out["schedule_coverage"],
            "defender_load": out["defender_load"],
            "schedule_entropy": out["schedule_entropy"],
        }


def build_defender_timing_schedule_network_from_config(config: dict[str, Any]) -> DefenderTimingScheduleNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return DefenderTimingScheduleNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        num_threats=int(cfg.get("num_threats", 6)),
        num_defenders=int(cfg.get("num_defenders", 6)),
        sinkhorn_iters=int(cfg.get("sinkhorn_iters", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
