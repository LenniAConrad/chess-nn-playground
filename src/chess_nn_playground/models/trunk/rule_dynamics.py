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
    _make_move_edges,
    _mlp,
)


class RuleConsistentLatentDynamics(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoder_channels: int = 96,
        latent_dim: int = 128,
        move_feature_dim: int = 32,
        max_moves: int = 32,
        max_invalid: int = 32,
        transition_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.max_moves = max_moves
        self.max_invalid = max_invalid
        self.encoder = SquareBoardEncoder(input_channels, encoder_channels, depth=2, latent_dim=latent_dim, dropout=dropout)
        edges = _make_move_edges()
        for key, tensor in edges.items():
            self.register_buffer(f"edge_{key}", tensor)
        self.move_type = nn.Embedding(4, move_feature_dim)
        self.distance_embedding = nn.Embedding(8, move_feature_dim)
        self.move_encoder = _mlp(encoder_channels * 2 + move_feature_dim, [latent_dim], latent_dim, dropout=dropout)
        self.legal_head = nn.Linear(latent_dim * 2, 1)
        self.transition = _mlp(latent_dim * 2, [latent_dim] * max(1, int(transition_layers)), latent_dim, dropout=dropout)
        self.head = _mlp(latent_dim + 5, [latent_dim], num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _board, squares, z = self.encoder(x)
        edge_count = min(self.max_moves + self.max_invalid, self.edge_src.numel())
        src = self.edge_src[:edge_count]
        dst = self.edge_dst[:edge_count]
        src_emb = squares.index_select(1, src)
        dst_emb = squares.index_select(1, dst)
        geom = self.move_type(self.edge_type[:edge_count]) + self.distance_embedding(self.edge_distance[:edge_count].clamp_max(7))
        move_tokens = self.move_encoder(torch.cat([src_emb, dst_emb, geom.to(dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)], dim=2))
        z_exp = z.unsqueeze(1).expand(-1, edge_count, -1)
        legal_logits = self.legal_head(torch.cat([z_exp, move_tokens], dim=2)).squeeze(-1)
        next_latents = self.transition(torch.cat([z_exp, move_tokens], dim=2))
        legal_prob = torch.sigmoid(legal_logits[:, : self.max_moves])
        transition_norm = (next_latents[:, : self.max_moves] - z.unsqueeze(1)).norm(dim=2)
        variance = next_latents[:, : self.max_moves].var(dim=1).mean(dim=1)
        stats = torch.stack(
            [
                legal_prob.mean(dim=1),
                _entropy(torch.softmax(legal_logits[:, : self.max_moves], dim=1), dim=1),
                transition_norm.mean(dim=1),
                transition_norm.amax(dim=1),
                variance,
            ],
            dim=1,
        )
        logits = _as_logits(self.head(torch.cat([z, stats], dim=1)), self.num_classes)
        return {
            "logits": logits,
            "legal_entropy": stats[:, 1],
            "transition_variance": variance,
            "max_transition_norm": stats[:, 3],
        }


def build_rule_dynamics_from_config(config: dict[str, Any]) -> RuleConsistentLatentDynamics:
    cfg = _common_config(config)
    return RuleConsistentLatentDynamics(
        **cfg,
        encoder_channels=int(config.get("encoder_channels", 96)),
        latent_dim=int(config.get("latent_dim", 128)),
        move_feature_dim=int(config.get("move_feature_dim", 32)),
        max_moves=int(config.get("max_moves", 32)),
        max_invalid=int(config.get("max_invalid", 32)),
        transition_layers=int(config.get("transition_layers", 2)),
    )
