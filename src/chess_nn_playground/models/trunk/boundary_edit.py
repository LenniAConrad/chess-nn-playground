from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor

from chess_nn_playground.models.trunk._research_blocks import (
    SquareBoardEncoder,
    _as_logits,
    _common_config,
    _mlp,
)


class BoundaryEditLagrangianNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoder_channels: int = 96,
        latent_dim: int = 128,
        max_edits: int = 32,
        solver_steps: int = 4,
        edit_feature_dim: int = 32,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_edits = max_edits
        self.solver_steps = solver_steps
        self.encoder = SquareBoardEncoder(input_channels, encoder_channels, depth=2, latent_dim=latent_dim, dropout=dropout)
        self.base_head = nn.Linear(latent_dim, 1)
        self.edit_embedding = nn.Embedding(max_edits, edit_feature_dim)
        self.delta_head = _mlp(latent_dim + edit_feature_dim, [latent_dim], latent_dim, dropout=dropout)
        self.cost_head = _mlp(latent_dim + edit_feature_dim, [latent_dim // 2], 1, dropout=dropout)
        self.energy_head = nn.Linear(latent_dim, 1)
        self.final_head = _mlp(8, [latent_dim // 2], num_classes, dropout=dropout)

    def _solve(self, z: torch.Tensor, deltas: torch.Tensor, costs: torch.Tensor, target_sign: float) -> tuple[torch.Tensor, torch.Tensor]:
        benefit = target_sign * self.energy_head(deltas).squeeze(-1)
        alpha = torch.sigmoid(benefit - costs)
        for _step in range(max(self.solver_steps - 1, 0)):
            edited = z + torch.einsum("be,bed->bd", alpha, deltas)
            residual_score = target_sign * self.energy_head(edited).detach()
            alpha = torch.sigmoid(benefit + 0.1 * residual_score - costs)
        edited = z + torch.einsum("be,bed->bd", alpha, deltas)
        logit = self.energy_head(edited).view(-1)
        effort = (alpha * costs).sum(dim=1)
        energy = effort + F.softplus(-target_sign * logit)
        return alpha, energy

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, _squares, z = self.encoder(x)
        edit_ids = torch.arange(self.max_edits, device=x.device)
        edit_features = self.edit_embedding(edit_ids).to(dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        context = z.unsqueeze(1).expand(-1, self.max_edits, -1)
        deltas = self.delta_head(torch.cat([context, edit_features], dim=2)) / math.sqrt(max(z.shape[1], 1))
        costs = F.softplus(self.cost_head(torch.cat([context, edit_features], dim=2)).squeeze(-1)) + 1e-3
        alpha_plus, e_plus = self._solve(z, deltas, costs, target_sign=1.0)
        alpha_minus, e_minus = self._solve(z, deltas, costs, target_sign=-1.0)
        base_logit = self.base_head(z).view(-1)
        edit_gap = e_minus - e_plus
        stats = torch.stack(
            [
                base_logit,
                e_plus,
                e_minus,
                edit_gap,
                alpha_plus.mean(dim=1),
                alpha_minus.mean(dim=1),
                (alpha_plus * costs).mean(dim=1),
                (alpha_minus * costs).mean(dim=1),
            ],
            dim=1,
        )
        logits = _as_logits(self.final_head(stats), self.num_classes)
        return {
            "logits": logits,
            "base_logit": base_logit,
            "E_plus": e_plus,
            "E_minus": e_minus,
            "edit_gap": edit_gap,
        }


def build_boundary_edit_from_config(config: dict[str, Any]) -> BoundaryEditLagrangianNetwork:
    cfg = _common_config(config)
    return BoundaryEditLagrangianNetwork(
        **cfg,
        encoder_channels=int(config.get("encoder_channels", 96)),
        latent_dim=int(config.get("latent_dim", 128)),
        max_edits=int(config.get("max_edits", 32)),
        solver_steps=int(config.get("solver_steps", 4)),
        edit_feature_dim=int(config.get("edit_feature_dim", 32)),
    )
