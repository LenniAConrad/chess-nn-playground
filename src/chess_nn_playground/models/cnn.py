from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class CNNConfig:
    input_channels: int = 18
    num_classes: int = 2
    channels: int = 64
    num_blocks: int = 3
    dropout: float = 0.1
    use_batchnorm: bool = True


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleChessCNN(nn.Module):
    """Small CNN baseline for board-plane tensors."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 2,
        channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        blocks: list[nn.Module] = []
        in_channels = input_channels
        for _idx in range(num_blocks):
            blocks.append(ConvBlock(in_channels, channels, use_batchnorm=use_batchnorm, dropout=dropout))
            in_channels = channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)
        self.config = CNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            num_blocks=num_blocks,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_cnn_from_config(config: dict[str, Any]) -> SimpleChessCNN:
    return SimpleChessCNN(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 2)),
        channels=int(config.get("channels", 64)),
        num_blocks=int(config.get("num_blocks", 3)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_summary_text(model: nn.Module) -> str:
    return f"{model}\n\nTrainable parameters: {count_parameters(model)}\n"
