"""ConvNeXt BoardNet implementation for idea i143."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class BoardCoordinatePlanes(nn.Module):
    """Deterministic 8x8 rank, file, center, and square-color planes."""

    coordinate_dim = 4

    def __init__(self) -> None:
        super().__init__()
        rank = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        center = 1.0 - 0.5 * (rank.abs() + file.abs())
        square = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8)
        color = ((square // 8 + square.remainder(8)).remainder(2.0) * 2.0) - 1.0
        self.register_buffer("planes", torch.cat([rank, file, center, color], dim=1), persistent=False)

    def for_batch(self, x: torch.Tensor) -> torch.Tensor:
        return self.planes.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.for_batch(x)], dim=1)


class ConvNeXtBoardBlock(nn.Module):
    """Depthwise board mixer followed by an inverted channel MLP."""

    def __init__(
        self,
        channels: int,
        mlp_hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        layer_scale_init: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if mlp_hidden_dim <= channels:
            raise ValueError("mlp_hidden_dim must be greater than channels for the inverted MLP")
        if kernel_size % 2 != 1 or kernel_size < 3:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1.0e-6)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden_dim, channels),
        )
        self.layer_scale = nn.Parameter(torch.full((channels,), float(layer_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        update = self.depthwise(x).permute(0, 2, 3, 1)
        update = self.norm(update)
        update = self.channel_mlp(update)
        update = update * self.layer_scale.view(1, 1, 1, -1)
        update = update.permute(0, 3, 1, 2).contiguous()
        return residual + update


class ConvNeXtBoardPoolingHead(nn.Module):
    """Global pooling head combining average, max, std, and learned attention pools."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        attention_hidden = max(8, channels // 2)
        self.attention = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, attention_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(attention_hidden, 1, kernel_size=1),
        )
        pooled_dim = channels * 4
        mid_dim = max(32, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat = x.flatten(2)
        mean_pool = flat.mean(dim=2)
        max_pool = flat.amax(dim=2)
        std_pool = flat.std(dim=2, unbiased=False)
        attention_logits = self.attention(x).flatten(1)
        attention = torch.softmax(attention_logits, dim=1)
        attention_pool = (flat * attention.unsqueeze(1)).sum(dim=2)
        pooled = torch.cat([mean_pool, max_pool, std_pool, attention_pool], dim=1)
        entropy = -(attention * attention.clamp_min(1.0e-8).log()).sum(dim=1) / math.log(float(attention.shape[1]))
        diagnostics = {
            "pool_attention_entropy": entropy,
            "pool_attention_peak": attention.amax(dim=1),
            "spatial_contrast": (max_pool - mean_pool).abs().mean(dim=1),
            "feature_std": std_pool.mean(dim=1),
        }
        return self.classifier(pooled), diagnostics


@dataclass(frozen=True)
class ConvNeXtBoardNetConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    mlp_hidden_dim: int = 96
    depth: int = 2
    kernel_size: int = 3
    dropout: float = 0.1
    layer_scale_init: float = 1.0e-3
    use_coordinate_planes: bool = True


class ConvNeXtBoardNet(nn.Module):
    """Small ConvNeXt-style classifier for 8x8 board-plane tensors."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        mlp_hidden_dim: int = 96,
        depth: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        layer_scale_init: float = 1.0e-3,
        use_coordinate_planes: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if mlp_hidden_dim <= channels:
            raise ValueError("mlp_hidden_dim must be greater than channels")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.use_coordinate_planes = bool(use_coordinate_planes)
        self.coordinates = BoardCoordinatePlanes() if self.use_coordinate_planes else None
        coordinate_dim = BoardCoordinatePlanes.coordinate_dim if self.use_coordinate_planes else 0
        stem_in = int(input_channels) + coordinate_dim
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, channels, kernel_size=3, padding=1),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBoardBlock(
                    channels=channels,
                    mlp_hidden_dim=mlp_hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = LayerNorm2d(channels)
        self.head = ConvNeXtBoardPoolingHead(
            channels=channels,
            hidden_dim=mlp_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.config = ConvNeXtBoardNetConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            mlp_hidden_dim=mlp_hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
            use_coordinate_planes=use_coordinate_planes,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        if self.coordinates is not None:
            augmented = self.coordinates(board)
        else:
            augmented = board
        features = self.stem(augmented)
        for block in self.blocks:
            features = block(features)
        features = self.final_norm(features)
        logits, diagnostics = self.head(features)
        output = {
            "logits": _format_logits(logits, self.num_classes),
            "convnext_feature_energy": features.square().mean(dim=(1, 2, 3)),
            "piece_density": board[:, : min(12, board.shape[1])].clamp(0.0, 1.0).sum(dim=1).clamp(0.0, 1.0).mean(dim=(1, 2)),
            "coordinate_response": self._coordinate_response(board),
        }
        output.update(diagnostics)
        return output

    def _coordinate_response(self, board: torch.Tensor) -> torch.Tensor:
        if self.coordinates is None:
            return board.new_zeros(board.shape[0])
        coord = self.coordinates.for_batch(board)
        stem_conv = self.stem[0]
        assert isinstance(stem_conv, nn.Conv2d)
        coord_weight = stem_conv.weight[:, -BoardCoordinatePlanes.coordinate_dim :, :, :]
        response = F.conv2d(coord, coord_weight, bias=None, padding=stem_conv.padding)
        return response.square().mean(dim=(1, 2, 3))


def build_convnext_boardnet_from_config(config: dict[str, Any]) -> ConvNeXtBoardNet:
    channels = int(config.get("channels", 64))
    mlp_hidden_dim = int(config.get("mlp_hidden_dim", config.get("hidden_dim", 96)))
    if mlp_hidden_dim <= channels:
        mlp_hidden_dim = channels + max(16, channels // 2)
    return ConvNeXtBoardNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=channels,
        mlp_hidden_dim=mlp_hidden_dim,
        depth=int(config.get("depth", config.get("num_blocks", 2))),
        kernel_size=int(config.get("kernel_size", 3)),
        dropout=float(config.get("dropout", 0.1)),
        layer_scale_init=float(config.get("layer_scale_init", 1.0e-3)),
        use_coordinate_planes=bool(config.get("use_coordinate_planes", config.get("coordinate_planes", True))),
    )
