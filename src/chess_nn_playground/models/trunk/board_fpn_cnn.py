"""Board FPN CNN implementation for idea i144."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.squeeze(-1) if num_classes == 1 else logits


class BoardFPNCoordinatePlanes(nn.Module):
    """Deterministic board-location planes for simple board tensors."""

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


class BoardFPNConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BoardFPNConvStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int = 2,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        layers: list[nn.Module] = []
        current = in_channels
        for _idx in range(blocks):
            layers.append(
                BoardFPNConvBlock(
                    current,
                    out_channels,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                )
            )
            current = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BoardFPNHead(nn.Module):
    def __init__(
        self,
        width: int,
        hidden_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pooled_dim = 2 * (width + 2 * width + 4 * width)
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

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)

    def forward(self, y8: torch.Tensor, y4: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([self._pool(y8), self._pool(y4), self._pool(x2)], dim=1))


@dataclass(frozen=True)
class BoardFPNCNNConfig:
    input_channels: int = 18
    num_classes: int = 1
    width: int = 48
    blocks_per_level: int = 2
    hidden_dim: int = 128
    dropout: float = 0.1
    use_batchnorm: bool = True
    use_coordinate_planes: bool = True
    ablation: str = "none"


class BoardFPNCNN(nn.Module):
    """Three-level feature-pyramid CNN over 8x8 chess board tensors."""

    VALID_ABLATIONS = {
        "none",
        "single_resolution_matched",
        "bottom_up_only",
        "no_2x2_level",
        "late_pool_only",
        "no_coordinate_planes",
    }

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        width: int = 48,
        blocks_per_level: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        use_coordinate_planes: bool = True,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown BoardFPNCNN ablation: {ablation}")
        if ablation == "no_coordinate_planes":
            use_coordinate_planes = False
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.width = int(width)
        self.ablation = ablation
        self.coordinates = BoardFPNCoordinatePlanes() if use_coordinate_planes else None
        coordinate_dim = BoardFPNCoordinatePlanes.coordinate_dim if use_coordinate_planes else 0
        self.level8 = BoardFPNConvStack(
            input_channels + coordinate_dim,
            width,
            blocks=blocks_per_level,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.level4 = BoardFPNConvStack(
            width,
            width * 2,
            blocks=blocks_per_level,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.level2 = BoardFPNConvStack(
            width * 2,
            width * 4,
            blocks=blocks_per_level,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.project2_to4 = nn.Conv2d(width * 4, width * 2, kernel_size=1)
        self.project4_to8 = nn.Conv2d(width * 2, width, kernel_size=1)
        self.head = BoardFPNHead(width=width, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
        self.config = BoardFPNCNNConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            width=width,
            blocks_per_level=blocks_per_level,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            use_coordinate_planes=use_coordinate_planes,
            ablation=ablation,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        augmented = self.coordinates(board) if self.coordinates is not None else board
        x8 = self.level8(augmented)
        x4 = self.level4(F.avg_pool2d(x8, kernel_size=2))
        x2 = self.level2(F.avg_pool2d(x4, kernel_size=2))

        y4 = self._fuse4(x4, x2)
        y8 = self._fuse8(x8, y4)
        head_x2 = x2
        if self.ablation == "single_resolution_matched":
            y4 = torch.zeros_like(y4)
            head_x2 = torch.zeros_like(head_x2)
        elif self.ablation == "no_2x2_level":
            head_x2 = torch.zeros_like(head_x2)
        logits = self.head(y8, y4, head_x2)
        return {
            "logits": _format_logits(logits, self.num_classes),
            "fpn_y8_energy": y8.square().mean(dim=(1, 2, 3)),
            "fpn_y4_energy": y4.square().mean(dim=(1, 2, 3)),
            "fpn_x2_energy": head_x2.square().mean(dim=(1, 2, 3)),
            "topdown_4_energy": self._topdown4_energy(x2),
            "topdown_8_energy": self._topdown8_energy(y4),
            "piece_density": board[:, : min(12, board.shape[1])].clamp(0.0, 1.0).sum(dim=1).clamp(0.0, 1.0).mean(dim=(1, 2)),
            "coordinate_energy": self._coordinate_energy(board),
        }

    def _fuse4(self, x4: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.ablation in {"bottom_up_only", "late_pool_only", "single_resolution_matched", "no_2x2_level"}:
            return x4
        topdown = F.interpolate(self.project2_to4(x2), size=x4.shape[-2:], mode="nearest")
        return x4 + topdown

    def _fuse8(self, x8: torch.Tensor, y4: torch.Tensor) -> torch.Tensor:
        if self.ablation in {"bottom_up_only", "late_pool_only"}:
            return x8
        if self.ablation == "single_resolution_matched":
            return x8
        topdown = F.interpolate(self.project4_to8(y4), size=x8.shape[-2:], mode="nearest")
        return x8 + topdown

    def _topdown4_energy(self, x2: torch.Tensor) -> torch.Tensor:
        if self.ablation in {"bottom_up_only", "late_pool_only", "single_resolution_matched", "no_2x2_level"}:
            return x2.new_zeros(x2.shape[0])
        update = self.project2_to4(x2)
        return update.square().mean(dim=(1, 2, 3))

    def _topdown8_energy(self, y4: torch.Tensor) -> torch.Tensor:
        if self.ablation in {"bottom_up_only", "late_pool_only", "single_resolution_matched"}:
            return y4.new_zeros(y4.shape[0])
        update = self.project4_to8(y4)
        return update.square().mean(dim=(1, 2, 3))

    def _coordinate_energy(self, board: torch.Tensor) -> torch.Tensor:
        if self.coordinates is None:
            return board.new_zeros(board.shape[0])
        coords = self.coordinates.for_batch(board)
        return coords.square().mean(dim=(1, 2, 3))


def build_board_fpn_cnn_from_config(config: dict[str, Any]) -> BoardFPNCNN:
    width = int(config.get("width", config.get("channels", 48)))
    return BoardFPNCNN(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        width=width,
        blocks_per_level=int(config.get("blocks_per_level", config.get("depth", 2))),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        use_coordinate_planes=bool(config.get("use_coordinate_planes", True)),
        ablation=str(config.get("ablation", "none")),
    )
