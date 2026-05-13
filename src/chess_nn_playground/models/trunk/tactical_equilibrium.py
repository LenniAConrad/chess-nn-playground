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


class TacticalEquilibriumNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 96,
        token_dim: int = 96,
        relation_dim: int = 32,
        max_attackers: int = 16,
        max_defenders: int = 24,
        solver_steps: int = 5,
        tau_attack: float = 0.7,
        tau_defense: float = 0.7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_attackers = max_attackers
        self.max_defenders = max_defenders
        self.solver_steps = solver_steps
        self.tau_attack = max(float(tau_attack), 1e-4)
        self.tau_defense = max(float(tau_defense), 1e-4)
        self.encoder = SquareBoardEncoder(input_channels, trunk_channels, depth=2, latent_dim=token_dim, dropout=dropout)
        self.register_buffer("square_geometry", _square_geometry())
        self.token_proj = _mlp(trunk_channels + 4, [token_dim], token_dim, dropout=dropout)
        self.attacker_selector = nn.Linear(token_dim, 1)
        self.defender_selector = nn.Linear(token_dim, 1)
        self.relation = _mlp(token_dim * 2 + 4, [relation_dim], relation_dim, dropout=dropout)
        self.payoff = _mlp(token_dim * 2 + relation_dim, [token_dim], 1, dropout=dropout)
        self.head = _mlp(token_dim + 8, [token_dim], num_classes, dropout=dropout)

    def _select(self, tokens: torch.Tensor, selector: nn.Linear, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = selector(tokens).squeeze(-1)
        values, indices = torch.topk(scores, k=min(k, tokens.shape[1]), dim=1)
        selected = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return selected, values, indices

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, squares, context = self.encoder(x)
        geom = self.square_geometry.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        tokens = self.token_proj(torch.cat([squares, geom], dim=2))
        attackers, attacker_scores, attacker_idx = self._select(tokens, self.attacker_selector, self.max_attackers)
        defenders, defender_scores, defender_idx = self._select(tokens, self.defender_selector, self.max_defenders)
        attacker_geom = torch.gather(geom, 1, attacker_idx.unsqueeze(-1).expand(-1, -1, geom.shape[-1]))
        defender_geom = torch.gather(geom, 1, defender_idx.unsqueeze(-1).expand(-1, -1, geom.shape[-1]))
        a = attackers.unsqueeze(2).expand(-1, -1, defenders.shape[1], -1)
        d = defenders.unsqueeze(1).expand(-1, attackers.shape[1], -1, -1)
        relation = self.relation(torch.cat([a, d, (attacker_geom.unsqueeze(2) - defender_geom.unsqueeze(1)).abs()], dim=3))
        payoff = self.payoff(torch.cat([a, d, relation], dim=3)).squeeze(-1)
        p = torch.softmax(attacker_scores, dim=1)
        q = torch.softmax(defender_scores, dim=1)
        for _step in range(max(self.solver_steps, 1)):
            p = torch.softmax(torch.bmm(payoff, q.unsqueeze(2)).squeeze(2) / self.tau_attack, dim=1)
            q = torch.softmax(-torch.bmm(payoff.transpose(1, 2), p.unsqueeze(2)).squeeze(2) / self.tau_defense, dim=1)
        value = torch.bmm(torch.bmm(p.unsqueeze(1), payoff), q.unsqueeze(2)).view(-1)
        attacker_best = payoff.mean(dim=2).amax(dim=1)
        defender_best = payoff.mean(dim=1).amin(dim=1)
        exploitability = (attacker_best - value).relu() + (value - defender_best).relu()
        stats = torch.stack(
            [
                value,
                _entropy(p, dim=1),
                _entropy(q, dim=1),
                exploitability,
                payoff.mean(dim=(1, 2)),
                payoff.amax(dim=(1, 2)),
                payoff.amin(dim=(1, 2)),
                attacker_scores.mean(dim=1) - defender_scores.mean(dim=1),
            ],
            dim=1,
        )
        logits = _as_logits(self.head(torch.cat([context, stats], dim=1)), self.num_classes)
        return {
            "logits": logits,
            "equilibrium_value": value,
            "attacker_entropy": stats[:, 1],
            "defender_entropy": stats[:, 2],
            "exploitability": exploitability,
        }


def build_tactical_equilibrium_from_config(config: dict[str, Any]) -> TacticalEquilibriumNetwork:
    cfg = _common_config(config)
    return TacticalEquilibriumNetwork(
        **cfg,
        trunk_channels=int(config.get("trunk_channels", 96)),
        token_dim=int(config.get("token_dim", 96)),
        relation_dim=int(config.get("relation_dim", 32)),
        max_attackers=int(config.get("max_attackers", 16)),
        max_defenders=int(config.get("max_defenders", 24)),
        solver_steps=int(config.get("solver_steps", 5)),
        tau_attack=float(config.get("tau_attack", 0.7)),
        tau_defense=float(config.get("tau_defense", 0.7)),
    )
