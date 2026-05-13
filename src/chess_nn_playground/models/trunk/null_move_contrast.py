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


class NullMoveContrastPuzzleNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        encoder_channels: int = 96,
        latent_dim: int = 128,
        pair_mixer_layers: int = 2,
        positive_null_margin: float = 0.5,
        dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = num_classes
        self.positive_null_margin = float(positive_null_margin)
        self.encoder = SquareBoardEncoder(input_channels, encoder_channels, depth=2, latent_dim=latent_dim, dropout=dropout)
        self.evidence_head = nn.Linear(latent_dim, 1)
        hidden = [latent_dim] * max(1, int(pair_mixer_layers))
        self.pair_mixer = _mlp(latent_dim * 4, hidden, latent_dim, dropout=dropout)
        self.head = _mlp(latent_dim + 4, [latent_dim], num_classes, dropout=dropout)

    def null_view(self, x: torch.Tensor) -> torch.Tensor:
        view = x.clone()
        if x.shape[1] > 12:
            view[:, 12:13] = 1.0 - view[:, 12:13]
        return view

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        _cur_board, _cur_squares, cur = self.encoder(x)
        _null_board, _null_squares, null = self.encoder(self.null_view(x))
        e_cur = self.evidence_head(cur)
        e_null = self.evidence_head(null)
        delta = e_cur - e_null
        pair = self.pair_mixer(torch.cat([cur, null, cur - null, cur * null], dim=1))
        logits = _as_logits(self.head(torch.cat([pair, e_cur, e_null, delta, delta.abs()], dim=1)), self.num_classes)
        return {
            "logits": logits,
            "current_evidence": e_cur.view(-1),
            "null_evidence": e_null.view(-1),
            "tempo_contrast_delta": delta.view(-1),
            "positive_null_margin": F.relu(self.positive_null_margin - delta.view(-1)),
        }


def build_null_move_contrast_from_config(config: dict[str, Any]) -> NullMoveContrastPuzzleNetwork:
    cfg = _common_config(config)
    return NullMoveContrastPuzzleNetwork(
        **cfg,
        encoder_channels=int(config.get("encoder_channels", 96)),
        latent_dim=int(config.get("latent_dim", 128)),
        pair_mixer_layers=int(config.get("pair_mixer_layers", 2)),
        positive_null_margin=float(config.get("positive_null_margin", 0.5)),
    )
