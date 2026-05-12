"""Tactical Effective Resistance Network for idea i209.

Implements the ``effective resistance over the tactical graph'' thesis:
the network builds a soft graph Laplacian on the 64 board squares from
learned threat affinities, runs a few power iterations of a damped
inverse, and reports an attacker-target effective-resistance proxy plus
spread statistics. The architecture is materially distinct from the
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


def _adjacency_prior() -> torch.Tensor:
    rank = torch.arange(64) // 8
    file = torch.arange(64) % 8
    dr = (rank.view(64, 1) - rank.view(1, 64)).float()
    df = (file.view(64, 1) - file.view(1, 64)).float()
    distance = (dr.square() + df.square()).clamp_min(1.0).sqrt()
    weights = torch.exp(-distance / 3.0)
    weights = weights - torch.diag(weights.diagonal())
    return weights


class GraphResistanceProbe(nn.Module):
    def __init__(self, channels: int, edge_dim: int, num_iterations: int = 6) -> None:
        super().__init__()
        self.edge_proj = nn.Linear(channels, edge_dim)
        self.endpoint_proj = nn.Linear(channels, 2)
        self.num_iterations = int(num_iterations)
        self.register_buffer("adjacency_prior", _adjacency_prior(), persistent=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        edge_emb = self.edge_proj(tokens)
        affinity = torch.einsum("bnd,bmd->bnm", edge_emb, edge_emb)
        affinity = torch.sigmoid(affinity) * self.adjacency_prior.unsqueeze(0)
        affinity = 0.5 * (affinity + affinity.transpose(1, 2))
        degree = affinity.sum(dim=-1).clamp_min(1.0e-3)
        endpoints = F.softmax(self.endpoint_proj(tokens), dim=1)
        source = endpoints[:, :, 0]
        target = endpoints[:, :, 1]
        b_vector = source - target
        alpha = torch.sigmoid(self.alpha)
        x = b_vector / degree
        for _ in range(self.num_iterations):
            propagated = torch.bmm(affinity, x.unsqueeze(-1)).squeeze(-1) / degree
            x = (1.0 - alpha) * x + alpha * (propagated + b_vector / degree)
        potential = x
        effective_resistance = (potential * b_vector).sum(dim=1)
        spread = potential.var(dim=1, unbiased=False)
        gradient_field = (affinity * (potential.unsqueeze(2) - potential.unsqueeze(1)).square()).sum(dim=(1, 2))
        return {
            "effective_resistance": effective_resistance,
            "potential": potential,
            "potential_spread": spread,
            "gradient_energy": gradient_field,
            "source_distribution": source,
            "target_distribution": target,
            "graph_degree_mean": degree.mean(dim=1),
        }


class TacticalEffectiveResistanceNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        edge_dim: int = 16,
        num_iterations: int = 6,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("TacticalEffectiveResistanceNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.probe = GraphResistanceProbe(channels, edge_dim, num_iterations)
        self.head_norm = nn.LayerNorm(8)
        self.head = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)
        tokens = feats.flatten(2).transpose(1, 2)
        probe = self.probe(tokens)
        readout = torch.stack(
            [
                probe["effective_resistance"],
                probe["potential_spread"],
                probe["gradient_energy"],
                probe["graph_degree_mean"],
                probe["potential"].abs().amax(dim=1),
                probe["potential"].abs().mean(dim=1),
                probe["source_distribution"].amax(dim=1),
                probe["target_distribution"].amax(dim=1),
            ],
            dim=1,
        )
        readout = self.head_norm(readout)
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "effective_resistance": probe["effective_resistance"],
            "potential_spread": probe["potential_spread"],
            "gradient_energy": probe["gradient_energy"],
            "graph_degree_mean": probe["graph_degree_mean"],
            "source_concentration": probe["source_distribution"].amax(dim=1),
            "target_concentration": probe["target_distribution"].amax(dim=1),
        }


def build_tactical_effective_resistance_network_from_config(config: dict[str, Any]) -> TacticalEffectiveResistanceNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return TacticalEffectiveResistanceNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        edge_dim=int(cfg.get("edge_dim", 16)),
        num_iterations=int(cfg.get("num_iterations", 6)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
