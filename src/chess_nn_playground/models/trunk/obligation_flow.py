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
    _entropy,
    _mlp,
    _square_geometry,
)


class PuzzleObligationFlowNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 96,
        token_dim: int = 96,
        max_obligations: int = 32,
        max_resources: int = 48,
        solver_steps: int = 4,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_obligations = max_obligations
        self.max_resources = max_resources
        self.solver_steps = solver_steps
        self.encoder = SquareBoardEncoder(input_channels, trunk_channels, depth=2, latent_dim=token_dim, dropout=dropout)
        self.register_buffer("square_geometry", _square_geometry())
        self.obligation_type = nn.Embedding(6, token_dim)
        self.resource_type = nn.Embedding(6, token_dim)
        self.token_proj = nn.Linear(trunk_channels + 4, token_dim)
        self.obligation_selector = nn.Linear(token_dim, 1)
        self.resource_selector = nn.Linear(token_dim, 1)
        self.demand_head = nn.Linear(token_dim, 1)
        self.capacity_head = nn.Linear(token_dim, 1)
        self.compatibility = nn.Bilinear(token_dim, token_dim, 1)
        self.head = _mlp(token_dim + 7, [token_dim], num_classes, dropout=dropout)

    def _select(self, tokens: torch.Tensor, selector: nn.Linear, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = selector(tokens).squeeze(-1)
        values, indices = torch.topk(scores, k=min(k, tokens.shape[1]), dim=1)
        selected = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return selected, values, indices

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, squares, context = self.encoder(x)
        geom = self.square_geometry.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        base_tokens = self.token_proj(torch.cat([squares, geom], dim=2))
        obligations, obligation_scores, obligation_idx = self._select(base_tokens, self.obligation_selector, self.max_obligations)
        resources, resource_scores, resource_idx = self._select(base_tokens, self.resource_selector, self.max_resources)
        obligation_types = (obligation_idx % 6).clamp_min(0)
        resource_types = (resource_idx % 6).clamp_min(0)
        obligations = obligations + self.obligation_type(obligation_types)
        resources = resources + self.resource_type(resource_types)
        demand = F.softplus(self.demand_head(obligations).squeeze(-1)) + 1e-3
        capacity = F.softplus(self.capacity_head(resources).squeeze(-1)) + 1e-3
        o_exp = obligations.unsqueeze(2).expand(-1, -1, resources.shape[1], -1)
        r_exp = resources.unsqueeze(1).expand(-1, obligations.shape[1], -1, -1)
        compatibility = self.compatibility(o_exp, r_exp).squeeze(-1)
        kernel = torch.exp(compatibility.clamp(-8.0, 8.0))
        allocation = kernel
        for _step in range(max(self.solver_steps, 1)):
            allocation = allocation * (demand.unsqueeze(2) / allocation.sum(dim=2, keepdim=True).clamp_min(1e-6))
            allocation = allocation * torch.minimum(
                torch.ones_like(allocation),
                capacity.unsqueeze(1) / allocation.sum(dim=1, keepdim=True).clamp_min(1e-6),
            )
        covered = allocation.sum(dim=2)
        residual = F.relu(demand - covered)
        allocation_prob = allocation / allocation.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        stats = torch.stack(
            [
                residual.mean(dim=1),
                residual.amax(dim=1),
                demand.mean(dim=1),
                capacity.mean(dim=1),
                compatibility.mean(dim=(1, 2)),
                _entropy(allocation_prob.flatten(1), dim=1),
                (obligation_scores.mean(dim=1) - resource_scores.mean(dim=1)),
            ],
            dim=1,
        )
        logits = _as_logits(self.head(torch.cat([context, stats], dim=1)), self.num_classes)
        return {
            "logits": logits,
            "flow_residual_mean": residual.mean(dim=1),
            "flow_residual_max": residual.amax(dim=1),
            "allocation_entropy": stats[:, 5],
        }


def build_obligation_flow_from_config(config: dict[str, Any]) -> PuzzleObligationFlowNetwork:
    cfg = _common_config(config)
    return PuzzleObligationFlowNetwork(
        **cfg,
        trunk_channels=int(config.get("trunk_channels", 96)),
        token_dim=int(config.get("token_dim", 96)),
        max_obligations=int(config.get("max_obligations", 32)),
        max_resources=int(config.get("max_resources", 48)),
        solver_steps=int(config.get("solver_steps", 4)),
    )
