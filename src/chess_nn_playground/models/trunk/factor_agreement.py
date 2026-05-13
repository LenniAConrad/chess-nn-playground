from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor

from chess_nn_playground.models.trunk._research_blocks import (
    ChessOperatorBlock,
    SquareBoardEncoder,
    _as_logits,
    _common_config,
    _make_operator_bank,
    _mean_square_pool,
    _mlp,
    _square_geometry,
)


class FactorAgreementClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        branch_dim: int = 64,
        disagreement_alpha: float = 0.5,
        uncertainty_beta: float = 0.1,
        residual_init_scale: float = 0.01,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.disagreement_alpha = float(disagreement_alpha)
        self.uncertainty_beta = float(uncertainty_beta)
        self.grid = SquareBoardEncoder(input_channels, branch_dim, depth=2, latent_dim=branch_dim, dropout=dropout)
        self.piece_proj = _mlp(12 + 4, [branch_dim], branch_dim, dropout=dropout)
        self.relation_ops = nn.ModuleList([ChessOperatorBlock(branch_dim, 5, dropout)])
        self.register_buffer("operator_bank", _make_operator_bank(["rank_ray", "file_ray", "diagonal_ray", "knight", "king"]))
        self.relation_proj = nn.Linear(input_channels + 4, branch_dim)
        self.global_proj = _mlp(input_channels + 8, [branch_dim], branch_dim, dropout=dropout)
        self.register_buffer("square_geometry", _square_geometry())
        self.evidence_heads = nn.ModuleList([nn.Linear(branch_dim, 1) for _ in range(4)])
        self.uncertainty_heads = nn.ModuleList([nn.Linear(branch_dim, 1) for _ in range(4)])
        self.residual = _mlp(branch_dim * 4, [branch_dim], num_classes, dropout=dropout)
        with torch.no_grad():
            for module in self.residual.modules():
                if isinstance(module, nn.Linear):
                    module.weight.mul_(float(residual_init_scale))
                    if module.bias is not None:
                        module.bias.mul_(float(residual_init_scale))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _grid_board, _grid_squares, grid_context = self.grid(x)
        geometry = self.square_geometry.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        square_planes = x.flatten(2).transpose(1, 2)
        piece_tokens = torch.cat([square_planes[:, :, :12], geometry], dim=2)
        occupancy = x[:, :12].sum(dim=1).flatten(1).clamp(0.0, 1.0) if x.shape[1] >= 12 else None
        piece_context = _mean_square_pool(self.piece_proj(piece_tokens), occupancy)
        relation = self.relation_proj(torch.cat([square_planes, geometry], dim=2))
        relation, _gates = self.relation_ops[0](relation, self.operator_bank)
        relation_context = relation.mean(dim=1)
        global_stats = torch.cat(
            [
                x.mean(dim=(2, 3)),
                x[:, :12].sum(dim=(2, 3)) if x.shape[1] >= 12 else torch.zeros(x.shape[0], 12, device=x.device, dtype=x.dtype),
            ],
            dim=1,
        )
        if global_stats.shape[1] < self.global_proj[0].in_features:
            global_stats = F.pad(global_stats, (0, self.global_proj[0].in_features - global_stats.shape[1]))
        global_context = self.global_proj(global_stats[:, : self.global_proj[0].in_features])
        factors = [grid_context, piece_context, relation_context, global_context]
        evidence = torch.cat([head(factor) for head, factor in zip(self.evidence_heads, factors)], dim=1)
        uncertainty = torch.cat([F.softplus(head(factor)) for head, factor in zip(self.uncertainty_heads, factors)], dim=1)
        mean_evidence = evidence.mean(dim=1, keepdim=True)
        disagreement = (evidence - mean_evidence).pow(2).mean(dim=1, keepdim=True)
        mean_uncertainty = uncertainty.mean(dim=1, keepdim=True)
        residual = self.residual(torch.cat(factors, dim=1))
        logits = mean_evidence - self.disagreement_alpha * disagreement - self.uncertainty_beta * mean_uncertainty + residual
        return {
            "logits": _as_logits(logits, self.num_classes),
            "factor_disagreement": disagreement.view(-1),
            "factor_uncertainty": mean_uncertainty.view(-1),
            "grid_evidence": evidence[:, 0],
            "piece_evidence": evidence[:, 1],
            "relation_evidence": evidence[:, 2],
            "global_evidence": evidence[:, 3],
        }


def build_factor_agreement_from_config(config: dict[str, Any]) -> FactorAgreementClassifier:
    cfg = _common_config(config)
    return FactorAgreementClassifier(
        **cfg,
        branch_dim=int(config.get("branch_dim", 64)),
        disagreement_alpha=float(config.get("disagreement_alpha", 0.5)),
        uncertainty_beta=float(config.get("uncertainty_beta", 0.1)),
        residual_init_scale=float(config.get("residual_init_scale", 0.01)),
    )
