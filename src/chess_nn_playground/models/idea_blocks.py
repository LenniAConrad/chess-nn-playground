from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BoardTensorSpec:
    input_channels: int = 18
    height: int = 8
    width: int = 8


def require_board_tensor(x: torch.Tensor, spec: BoardTensorSpec) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected board tensor with shape (batch, channels, 8, 8), got {tuple(x.shape)}")
    expected = (spec.input_channels, spec.height, spec.width)
    actual = tuple(x.shape[1:])
    if actual != expected:
        raise ValueError(f"Expected board tensor tail shape {expected}, got {actual}")
    return x


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, use_batchnorm: bool = True) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BoardConvStem(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        depth: int = 2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(depth):
            layers.append(ConvNormAct(in_channels, channels, use_batchnorm=use_batchnorm))
            in_channels = channels
        self.layers = nn.Sequential(*layers)
        self.output_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(require_board_tensor(x, self.spec))


class GlobalPoolClassifier(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(input_channels, num_classes))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(x))
