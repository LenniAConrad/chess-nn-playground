"""Baseline conv spatial mixer -- reproduces the original lc0_bt4 block's mixing.

This is the control: bt4_primitive_mixer with mixer="conv" should behave like
the repo's lc0_bt4_classifier residual tower.
"""

from __future__ import annotations

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers._base import register_mixer


class ConvMixer(nn.Module):
    def __init__(self, channels: int, use_batchnorm: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        return y


@register_mixer("conv")
def build(channels: int, use_batchnorm: bool = True, dropout: float = 0.1, **_: object) -> nn.Module:
    return ConvMixer(channels=channels, use_batchnorm=use_batchnorm, dropout=dropout)
