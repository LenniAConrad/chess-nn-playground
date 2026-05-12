from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ResidualCNNConfig:
    input_channels: int = 18
    num_classes: int = 2
    channels: int = 64
    num_blocks: int = 6
    dropout: float = 0.1
    use_batchnorm: bool = True


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        self.body = nn.Sequential(*layers)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class ResidualChessCNN(nn.Module):
    """Small residual CNN reference architecture for board-plane classification."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 2,
        channels: int = 64,
        num_blocks: int = 6,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, use_batchnorm=use_batchnorm, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )
        self.config = ResidualCNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            num_blocks=num_blocks,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def build_residual_cnn_from_config(config: dict[str, Any]) -> ResidualChessCNN:
    return ResidualChessCNN(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 2)),
        channels=int(config.get("channels", 64)),
        num_blocks=int(config.get("num_blocks", 6)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
