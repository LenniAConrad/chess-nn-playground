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


@dataclass
class MLPConfig:
    input_channels: int = 18
    num_classes: int = 2
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    use_layernorm: bool = True


class BoardMLP(nn.Module):
    """Flattened board-plane MLP baseline."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 2,
        hidden_dims: Sequence[int] = (512, 256, 128),
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        dims = [input_channels * 8 * 8, *[int(dim) for dim in hidden_dims]]
        if len(dims) < 2:
            raise ValueError("hidden_dims must contain at least one layer")

        layers: list[nn.Module] = [nn.Flatten()]
        for in_features, out_features in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_features, out_features))
            if use_layernorm:
                layers.append(nn.LayerNorm(out_features))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))

        self.network = nn.Sequential(*layers)
        self.config = MLPConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            hidden_dims=tuple(int(dim) for dim in hidden_dims),
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_mlp_from_config(config: dict[str, Any]) -> BoardMLP:
    return BoardMLP(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 2)),
        hidden_dims=_hidden_dims(config.get("hidden_dims"), (512, 256, 128)),
        dropout=float(config.get("dropout", 0.1)),
        use_layernorm=bool(config.get("use_layernorm", True)),
    )
