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
    DeterministicTacticalMaskBuilder,
)


class MaterialLockedTacticalDROClassifier(nn.Module):
    """Classifier with bounded adversarial contamination over deterministic tactical masks."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 96,
        depth: int = 3,
        hidden_dim: int = 128,
        mask_channels: int = 10,
        rho: float = 0.08,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("MaterialLockedTacticalDROClassifier supports puzzle_binary single-logit output")
        self.encoder = CompactBoardEncoder(input_channels, channels, depth, hidden_dim, dropout, use_batchnorm)
        self.mask_builder = DeterministicTacticalMaskBuilder()
        self.mask_projection = nn.Sequential(
            nn.Conv2d(mask_channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.delta_projection = nn.Sequential(
            nn.Conv2d(mask_channels, channels, kernel_size=1),
            nn.Tanh(),
        )
        self.clean_head = nn.Sequential(
            nn.Linear(channels * 2 + 17, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.rho = float(rho)
        self.spec = BoardTensorSpec(input_channels=input_channels)

    def _pool(self, board: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return torch.cat([board.mean(dim=(2, 3)), board.amax(dim=(2, 3)), stats], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        base_board, _hidden, stats = self.encoder(x)
        masks = self.mask_builder(x)
        support = (masks > 0).to(dtype=x.dtype)
        clean_board = base_board + self.mask_projection(masks)
        delta = self.rho * support.mean(dim=1, keepdim=True) * self.delta_projection(masks)
        adversarial_board = clean_board + delta
        clean_logits = self.clean_head(self._pool(clean_board, stats)).view(-1)
        adversarial_logits = self.clean_head(self._pool(adversarial_board, stats)).view(-1)
        budget_used = delta.abs().mean(dim=(1, 2, 3)) / max(self.rho, 1e-6)
        return {
            "logits": clean_logits,
            "clean_logits": clean_logits,
            "adversarial_logits": adversarial_logits,
            "tactical_mask_mean": masks.mean(dim=(1, 2, 3)),
            "mask_budget_used": budget_used,
            "material_total": stats[:, -1],
            "material_delta": stats[:, -2],
        }


def build_material_locked_tactical_dro_from_config(config: dict[str, Any]) -> MaterialLockedTacticalDROClassifier:
    return MaterialLockedTacticalDROClassifier(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 96)),
        depth=int(config.get("depth", 3)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        rho=float(config.get("rho", 0.08)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
