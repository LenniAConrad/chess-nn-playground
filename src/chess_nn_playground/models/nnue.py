from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn


def _hidden_dims(value: Any, default: Sequence[int]) -> list[int]:
    if value is None:
        return [int(item) for item in default]
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(",", " ").split()]
        return [int(item) for item in items if item]
    return [int(item) for item in value]


class ClippedReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=1.0)


@dataclass
class NNUEConfig:
    input_channels: int = 18
    num_classes: int = 2
    accumulator_size: int = 256
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.05
    include_state_planes: bool = True


class StockfishStyleNNUE(nn.Module):
    """Trainable Stockfish-style NNUE baseline for current-position board tensors.

    This keeps the NNUE idea that both side-to-move and opponent perspective
    accumulators are built before a small clipped-ReLU head. It is not a loader
    for Stockfish production NNUE weights.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 2,
        accumulator_size: int = 256,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.05,
        include_state_planes: bool = True,
    ) -> None:
        super().__init__()
        if input_channels < 12:
            raise ValueError("StockfishStyleNNUE expects at least 12 piece planes")
        if accumulator_size <= 0:
            raise ValueError("accumulator_size must be positive")

        self.input_channels = int(input_channels)
        self.include_state_planes = bool(include_state_planes)
        state_dims = max(0, input_channels - 12) if include_state_planes else 0

        self.feature_transform = nn.Linear(12 * 8 * 8, accumulator_size)
        self.accumulator_activation = ClippedReLU()

        head_dims = [2 * accumulator_size + state_dims, *[int(dim) for dim in hidden_dims]]
        if len(head_dims) < 2:
            raise ValueError("hidden_dims must contain at least one layer")
        head: list[nn.Module] = []
        for in_features, out_features in zip(head_dims, head_dims[1:]):
            head.append(nn.Linear(in_features, out_features))
            head.append(ClippedReLU())
            if dropout > 0:
                head.append(nn.Dropout(dropout))
        head.append(nn.Linear(head_dims[-1], num_classes))
        self.head = nn.Sequential(*head)

        self.config = NNUEConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            accumulator_size=accumulator_size,
            hidden_dims=tuple(int(dim) for dim in hidden_dims),
            dropout=dropout,
            include_state_planes=include_state_planes,
        )

    def _perspective_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pieces = x[:, :12]
        white_perspective = pieces
        black_perspective = torch.cat(
            [
                torch.flip(pieces[:, 6:12], dims=(-2, -1)),
                torch.flip(pieces[:, 0:6], dims=(-2, -1)),
            ],
            dim=1,
        )
        return white_perspective.reshape(x.shape[0], -1), black_perspective.reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        white_features, black_features = self._perspective_features(x)
        white_acc = self.accumulator_activation(self.feature_transform(white_features))
        black_acc = self.accumulator_activation(self.feature_transform(black_features))

        if x.shape[1] > 12:
            white_to_move = x[:, 12:13].mean(dim=(-2, -1)).clamp(0.0, 1.0)
        else:
            white_to_move = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        black_to_move = 1.0 - white_to_move
        stm_acc = white_to_move * white_acc + black_to_move * black_acc
        opp_acc = white_to_move * black_acc + black_to_move * white_acc

        parts = [stm_acc, opp_acc]
        if self.include_state_planes and x.shape[1] > 12:
            parts.append(x[:, 12:self.input_channels].mean(dim=(-2, -1)))
        return self.head(torch.cat(parts, dim=1))


def build_nnue_from_config(config: dict[str, Any]) -> StockfishStyleNNUE:
    return StockfishStyleNNUE(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 2)),
        accumulator_size=int(config.get("accumulator_size", 256)),
        hidden_dims=_hidden_dims(config.get("hidden_dims"), (128, 64)),
        dropout=float(config.get("dropout", 0.05)),
        include_state_planes=bool(config.get("include_state_planes", True)),
    )
