"""BT4-style residual tower with a swappable per-block spatial mixer.

The repo's `lc0_bt4_classifier` is a residual conv tower: each block mixes the
8x8 board spatially with a pair of 3x3 convs, then SqueezeExcite + residual.
This model keeps that tower shell (stem -> N blocks -> value head) but makes
the per-block spatial-mixing operator swappable via a `mixer` config field.

- `mixer: conv`       reproduces the original lc0_bt4 block (control).
- `mixer: attention`  generic multi-head self-attention over the 64 squares.
- `mixer: <primitive>` a chess-aware primitive used as the spatial mixer.

Every mixer obeys the contract `(B, C, 8, 8) -> (B, C, 8, 8)` (see
`bt4_mixers/_base.py`). The block wraps it with SqueezeExcite + residual +
activation, so swapping the mixer is the only architectural change between
runs -- the cleanest possible test of "is this primitive a better spatial
mixer than conv / attention?".
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.architecture.bt4_mixers import build_mixer


class _SqueezeExcite(nn.Module):
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


class BT4MixerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mixer_name: str,
        se_channels: int,
        use_batchnorm: bool,
        dropout: float,
        mixer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.mixer = build_mixer(
            mixer_name,
            channels=channels,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            **(mixer_kwargs or {}),
        )
        self.se = _SqueezeExcite(channels, se_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.mixer(x)
        if y.shape != x.shape:
            raise ValueError(
                f"mixer {type(self.mixer).__name__} returned shape {tuple(y.shape)}, "
                f"expected {tuple(x.shape)}"
            )
        y = self.se(y)
        return self.activation(residual + y)


class BT4PrimitiveMixerNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        num_blocks: int = 4,
        value_channels: int = 16,
        value_hidden: int = 128,
        se_channels: int = 16,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        mixer: str = "conv",
        mixer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_classes = int(num_classes)
        self.mixer_name = str(mixer)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                BT4MixerBlock(
                    channels=channels,
                    mixer_name=mixer,
                    se_channels=se_channels,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                    mixer_kwargs=mixer_kwargs,
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
            nn.Linear(value_hidden, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        logits = self.value_head(x)
        if self.num_classes == 1:
            logits = logits.view(-1)
        return {"logits": logits}


def build_bt4_primitive_mixer_from_config(config: dict[str, Any]) -> BT4PrimitiveMixerNet:
    mixer_kwargs = config.get("mixer_kwargs")
    if mixer_kwargs is not None and not isinstance(mixer_kwargs, dict):
        raise ValueError("mixer_kwargs must be a mapping if provided")
    return BT4PrimitiveMixerNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        num_blocks=int(config.get("num_blocks", config.get("depth", 4))),
        value_channels=int(config.get("value_channels", 16)),
        value_hidden=int(config.get("value_hidden", 128)),
        se_channels=int(config.get("se_channels", 16)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        mixer=str(config.get("mixer", "conv")),
        mixer_kwargs=mixer_kwargs,
    )
