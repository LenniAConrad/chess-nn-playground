"""Hierarchical Tactical Option Network for idea i203.

Implements the ``options as a hierarchy of subgoals`` thesis from the
batch-4 research packet: option proposals are generated from board
features, gated by a two-level hierarchical controller, and aggregated
into a single puzzle logit. The architecture is materially distinct from
the shared research-packet probe.
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


class OptionProposer(nn.Module):
    def __init__(self, channels: int, num_options: int, option_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_options = int(num_options)
        self.option_dim = int(option_dim)
        self.queries = nn.Parameter(torch.randn(num_options, channels) * 0.02)
        self.value = nn.Linear(channels, option_dim)
        self.score = nn.Linear(channels, num_options)
        self.dropout = nn.Dropout(dropout)

    def forward(self, board_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_logits = torch.einsum("bsc,oc->bos", board_tokens, self.queries)
        attn = F.softmax(attn_logits, dim=-1)
        values = self.value(board_tokens)
        options = torch.bmm(attn, values)
        salience = self.score(board_tokens).transpose(1, 2)
        option_salience = (attn * salience).sum(dim=-1)
        return self.dropout(options), option_salience


class HierarchicalGate(nn.Module):
    def __init__(self, num_options: int, option_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.parent_proj = nn.Linear(option_dim, hidden_dim)
        self.child_proj = nn.Linear(option_dim, hidden_dim)
        self.tree_norm = nn.LayerNorm(hidden_dim)
        self.utility = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, options: torch.Tensor, salience: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parent = F.softmax(salience, dim=1)
        parent_summary = (options * parent.unsqueeze(-1)).sum(dim=1, keepdim=True)
        children = self.parent_proj(parent_summary) + self.child_proj(options)
        children = self.tree_norm(F.gelu(self.dropout(children)))
        child_utility = self.utility(children).squeeze(-1)
        child_weights = F.softmax(child_utility, dim=1)
        terminal = (children * child_weights.unsqueeze(-1)).sum(dim=1)
        return terminal, parent, child_weights


class HierarchicalTacticalOptionNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_options: int = 8,
        option_dim: int = 32,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("HierarchicalTacticalOptionNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.proposer = OptionProposer(channels, num_options, option_dim, dropout)
        self.gate = HierarchicalGate(num_options, option_dim, hidden_dim, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 4),
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)
        tokens = feats.flatten(2).transpose(1, 2)
        options, salience = self.proposer(tokens)
        terminal, parent_weights, child_weights = self.gate(options, salience)
        gate_entropy = -(child_weights.clamp_min(1.0e-6).log() * child_weights).sum(dim=1)
        parent_entropy = -(parent_weights.clamp_min(1.0e-6).log() * parent_weights).sum(dim=1)
        option_norm = options.norm(dim=-1).mean(dim=1)
        salience_max = salience.amax(dim=1)
        readout = torch.cat(
            [terminal, parent_entropy.unsqueeze(-1), gate_entropy.unsqueeze(-1), option_norm.unsqueeze(-1), salience_max.unsqueeze(-1)],
            dim=-1,
        )
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "option_utilities": child_weights,
            "parent_option_weights": parent_weights,
            "hierarchical_gate_entropy": gate_entropy,
            "parent_entropy": parent_entropy,
            "option_norm": option_norm,
            "top_option_index": child_weights.argmax(dim=1).to(logits.dtype),
        }


def build_hierarchical_tactical_option_network_from_config(config: dict[str, Any]) -> HierarchicalTacticalOptionNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return HierarchicalTacticalOptionNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        num_options=int(cfg.get("num_options", 8)),
        option_dim=int(cfg.get("option_dim", 32)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
