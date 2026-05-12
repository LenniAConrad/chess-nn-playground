"""Pre-wired model skeleton for new idea folders.

Copy this file into an idea folder, rename the class, and replace the feature
body with the idea-specific math. Keep the input guard and classifier contract.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardConvStem, GlobalPoolClassifier


class TemplateIdeaModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        stem_depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = BoardConvStem(
            input_channels=input_channels,
            channels=channels,
            depth=stem_depth,
        )
        self.head = GlobalPoolClassifier(
            input_channels=self.stem.output_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stem(x))


def build_model_from_config(config: dict[str, Any]) -> TemplateIdeaModel:
    return TemplateIdeaModel(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        stem_depth=int(config.get("stem_depth", 2)),
        dropout=float(config.get("dropout", 0.0)),
    )
