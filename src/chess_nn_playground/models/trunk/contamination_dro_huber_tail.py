from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor

from chess_nn_playground.models.trunk._gpt_research_blocks import (
    CompactBoardEncoder,
    RobustBoardClassifierConfig,
)


class ContaminationDROHuberTailClassifier(nn.Module):
    """Single-logit board classifier trained with a near-puzzle robust tail loss."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        cfg = RobustBoardClassifierConfig(**{**RobustBoardClassifierConfig().__dict__, **kwargs})
        if cfg.num_classes != 1:
            raise ValueError("ContaminationDROHuberTailClassifier supports puzzle_binary single-logit output")
        self.encoder = CompactBoardEncoder(
            input_channels=cfg.input_channels,
            channels=cfg.channels,
            depth=cfg.depth,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm,
        )
        self.head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        _board, hidden, stats = self.encoder(x)
        logits = self.head(hidden).view(-1)
        return {
            "logits": logits,
            "material_total": stats[:, -1],
            "material_delta": stats[:, -2],
            "logit_margin_residual": F.relu(logits),
        }


def build_contamination_dro_huber_tail_from_config(config: dict[str, Any]) -> ContaminationDROHuberTailClassifier:
    return ContaminationDROHuberTailClassifier(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 96)),
        depth=int(config.get("depth", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
