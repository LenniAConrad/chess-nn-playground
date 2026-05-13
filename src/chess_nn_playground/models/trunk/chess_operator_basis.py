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
    _as_logits,
    _common_config,
    _entropy,
    _make_operator_bank,
    _mean_square_pool,
    _mlp,
    _square_geometry,
)


class ChessOperatorBasisClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        hidden_dim: int = 96,
        blocks: int = 4,
        relation_operators: Sequence[str] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        relation_operators = relation_operators or [
            "identity",
            "rank_ray",
            "file_ray",
            "diagonal_ray",
            "antidiagonal_ray",
            "knight",
            "king",
            "white_pawn_attack",
            "black_pawn_attack",
            "king_zone",
        ]
        self.register_buffer("operator_bank", _make_operator_bank(relation_operators))
        self.input_projection = nn.Sequential(
            nn.Linear(input_channels + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.register_buffer("square_geometry", _square_geometry())
        self.blocks = nn.ModuleList([ChessOperatorBlock(hidden_dim, len(relation_operators), dropout) for _ in range(blocks)])
        self.head = _mlp(hidden_dim * 3, [hidden_dim], num_classes, dropout=dropout)

    def _king_zone_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 12:
            return torch.ones(x.shape[0], 64, device=x.device, dtype=x.dtype)
        kings = x[:, [5, 11]].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        zone = F.max_pool2d(kings, kernel_size=3, stride=1, padding=1)
        return zone.flatten(1).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board_squares = x.flatten(2).transpose(1, 2)
        geometry = self.square_geometry.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        h = self.input_projection(torch.cat([board_squares, geometry], dim=2))
        gate_trace = []
        for block in self.blocks:
            h, gates = block(h, self.operator_bank)
            gate_trace.append(gates)
        occupancy = x[:, :12].sum(dim=1).flatten(1).clamp(0.0, 1.0) if x.shape[1] >= 12 else None
        pooled = h.mean(dim=1)
        piece_pooled = _mean_square_pool(h, occupancy)
        king_pooled = _mean_square_pool(h, self._king_zone_mask(x))
        logits = _as_logits(self.head(torch.cat([pooled, piece_pooled, king_pooled], dim=1)), self.num_classes)
        gates_all = torch.stack(gate_trace, dim=1)
        return {"logits": logits, "operator_gate_entropy": _entropy(gates_all.mean(dim=1), dim=1)}


def build_chess_operator_basis_from_config(config: dict[str, Any]) -> ChessOperatorBasisClassifier:
    cfg = _common_config(config)
    return ChessOperatorBasisClassifier(
        **cfg,
        hidden_dim=int(config.get("hidden_dim", 96)),
        blocks=int(config.get("blocks", config.get("num_blocks", 4))),
        relation_operators=config.get("relation_operators"),
    )
