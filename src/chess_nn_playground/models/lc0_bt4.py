from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class LC0BT4Config:
    input_channels: int = 112
    num_classes: int = 2
    channels: int = 64
    num_blocks: int = 4
    value_channels: int = 16
    value_hidden: int = 128
    se_channels: int = 16
    dropout: float = 0.1
    use_batchnorm: bool = True


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, se_channels: int) -> None:
        super().__init__()
        se_channels = max(1, int(se_channels))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, se_channels),
            nn.ReLU(inplace=True),
            nn.Linear(se_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x)).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class LC0BT4Block(nn.Module):
    def __init__(self, channels: int, se_channels: int, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.se = SqueezeExcite(channels, se_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.activation(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        y = self.se(y)
        return self.activation(residual + y)


class LC0BT4Classifier(nn.Module):
    """LC0 BT4-style residual tower adapted to puzzle-signal classification."""

    def __init__(
        self,
        input_channels: int = 112,
        num_classes: int = 2,
        channels: int = 64,
        num_blocks: int = 4,
        value_channels: int = 16,
        value_hidden: int = 128,
        se_channels: int = 16,
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
            *[
                LC0BT4Block(
                    channels=channels,
                    se_channels=se_channels,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_channels, kernel_size=1, bias=not use_batchnorm),
            nn.BatchNorm2d(value_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(value_channels * 8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(value_hidden, num_classes),
        )
        self.config = LC0BT4Config(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            num_blocks=num_blocks,
            value_channels=value_channels,
            value_hidden=value_hidden,
            se_channels=se_channels,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.value_head(x)


def build_lc0_bt4_from_config(config: dict[str, Any]) -> LC0BT4Classifier:
    return LC0BT4Classifier(
        input_channels=int(config.get("input_channels", 112)),
        num_classes=int(config.get("num_classes", 2)),
        channels=int(config.get("channels", 64)),
        num_blocks=int(config.get("num_blocks", 4)),
        value_channels=int(config.get("value_channels", 16)),
        value_hidden=int(config.get("value_hidden", 128)),
        se_channels=int(config.get("se_channels", 16)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
    )
