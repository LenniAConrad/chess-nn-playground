from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _norm(channels: int, use_batchnorm: bool) -> nn.Module:
    return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()


class ConvNormGelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_batchnorm),
            _norm(out_channels, use_batchnorm),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PyramidResidualBlock(nn.Module):
    def __init__(self, channels: int, *, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormGelu(channels, channels, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            _norm(channels, use_batchnorm),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class BoardSummary(nn.Module):
    summary_dim = 18

    def __init__(self) -> None:
        super().__init__()
        rank = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 8, 1)
        file = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 1, 8)
        rank_grid = rank.expand(1, 1, 8, 8)
        file_grid = file.expand(1, 1, 8, 8)
        center = (1.0 - torch.maximum(rank_grid.abs(), file_grid.abs())).clamp_min(0.0)
        values = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0, -1.0, -3.0, -3.0, -5.0, -9.0, 0.0])
        self.register_buffer("rank_grid", rank_grid, persistent=False)
        self.register_buffer("file_grid", file_grid, persistent=False)
        self.register_buffer("center_weight", center, persistent=False)
        self.register_buffer("piece_values", values, persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.shape[0]
        piece = x[:, : min(12, x.shape[1])].clamp(0.0, 1.0)
        if piece.shape[1] < 12:
            piece = F.pad(piece, (0, 0, 0, 0, 0, 12 - piece.shape[1]))
        counts = piece.flatten(2).sum(dim=2)
        side = x[:, 12:13].mean(dim=(2, 3)) if x.shape[1] > 12 else counts.new_zeros(batch, 1)
        white_to_move = side[:, 0]
        side_sign = white_to_move.mul(2.0).sub(1.0)
        material_raw = (counts * self.piece_values.to(dtype=x.dtype).view(1, 12)).sum(dim=1)
        material_balance = side_sign * material_raw / 39.0
        occupancy = piece.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        occupancy_count = occupancy.flatten(1).sum(dim=1) / 32.0
        denom = occupancy.flatten(1).sum(dim=1).clamp_min(1.0)
        rank_imbalance = (occupancy * self.rank_grid.to(dtype=x.dtype)).sum(dim=(1, 2, 3)) / denom
        file_imbalance = (occupancy * self.file_grid.to(dtype=x.dtype)).sum(dim=(1, 2, 3)) / denom
        center_pressure = (occupancy * self.center_weight.to(dtype=x.dtype)).sum(dim=(1, 2, 3)) / denom
        features = torch.cat(
            [
                counts / 8.0,
                side,
                material_balance.unsqueeze(1),
                occupancy_count.unsqueeze(1),
                rank_imbalance.unsqueeze(1),
                file_imbalance.unsqueeze(1),
                center_pressure.unsqueeze(1),
            ],
            dim=1,
        )
        return {
            "features": features,
            "material_balance": material_balance,
            "occupancy_count": occupancy_count,
            "rank_file_imbalance": torch.sqrt(rank_imbalance.square() + file_imbalance.square()),
            "center_pressure": center_pressure,
        }


@dataclass(frozen=True)
class CoarseToFineResidualPyramidConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True
    residual_scale: float = 1.0


class CoarseToFineBoardResidualPyramid(nn.Module):
    """Classify from detail left after coarse board reconstructions explain finer scales."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("CoarseToFineBoardResidualPyramid supports puzzle_binary single-logit output")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.summary = BoardSummary()
        self.stem = nn.Sequential(
            ConvNormGelu(input_channels + 2, channels, use_batchnorm=use_batchnorm, dropout=dropout),
            *[PyramidResidualBlock(channels, use_batchnorm=use_batchnorm, dropout=dropout) for _ in range(depth)],
        )
        self.down8_to4 = ConvNormGelu(channels, channels, use_batchnorm=use_batchnorm, dropout=dropout)
        self.down4_to2 = ConvNormGelu(channels, channels, use_batchnorm=use_batchnorm, dropout=dropout)
        self.decode2_to4 = nn.Sequential(
            ConvNormGelu(channels, channels, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.decode4_to8 = nn.Sequential(
            ConvNormGelu(channels, channels, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.residual4_refine = PyramidResidualBlock(channels, use_batchnorm=use_batchnorm, dropout=dropout)
        self.residual8_refine = PyramidResidualBlock(channels, use_batchnorm=use_batchnorm, dropout=dropout)
        self.coarse_head = nn.Sequential(
            nn.Linear(channels * 2 + BoardSummary.summary_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(channels * 4 + 10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Linear(hidden_dim * 2, 1)
        rank = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        self.register_buffer("coordinate_planes", torch.cat([rank, file], dim=1), persistent=False)
        self.residual_scale = float(residual_scale)
        self.config = CoarseToFineResidualPyramidConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            residual_scale=float(residual_scale),
        )

    @staticmethod
    def _pool_pair(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)

    @staticmethod
    def _residual_stats(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = residual.flatten(1)
        l1 = flat.abs().mean(dim=1)
        l2 = torch.sqrt(flat.square().mean(dim=1).clamp_min(1.0e-12))
        max_abs = flat.abs().amax(dim=1)
        return l1, l2, max_abs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        summary = self.summary(x)
        coords = self.coordinate_planes.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1)
        fine8 = self.stem(torch.cat([x, coords], dim=1))
        actual4 = self.down8_to4(F.avg_pool2d(fine8, kernel_size=2))
        coarse2 = self.down4_to2(F.avg_pool2d(actual4, kernel_size=2))

        predicted4 = F.interpolate(self.decode2_to4(coarse2), size=(4, 4), mode="bilinear", align_corners=False)
        residual4 = self.residual4_refine(actual4 - predicted4)
        explained4 = predicted4 + self.residual_scale * residual4
        predicted8 = F.interpolate(self.decode4_to8(explained4), size=(8, 8), mode="bilinear", align_corners=False)
        residual8 = self.residual8_refine(fine8 - predicted8)
        explained8 = predicted8 + self.residual_scale * residual8

        residual4_l1, residual4_l2, residual4_max = self._residual_stats(residual4)
        residual8_l1, residual8_l2, residual8_max = self._residual_stats(residual8)
        coarse_l2 = torch.sqrt(coarse2.flatten(1).square().mean(dim=1).clamp_min(1.0e-12))
        explained_l2 = torch.sqrt(explained8.flatten(1).square().mean(dim=1).clamp_min(1.0e-12))
        unexplained_ratio = residual8_l2 / (explained_l2 + residual8_l2).clamp_min(1.0e-6)
        residual_gain = residual8_l2 / residual4_l2.clamp_min(1.0e-6)
        detail_concentration = residual8_max / residual8_l1.clamp_min(1.0e-6)
        residual_alignment = F.cosine_similarity(
            F.interpolate(residual4, size=(8, 8), mode="bilinear", align_corners=False).flatten(1),
            residual8.flatten(1),
            dim=1,
        )

        coarse_features = self.coarse_head(torch.cat([self._pool_pair(coarse2), summary["features"]], dim=1))
        residual_features = torch.cat(
            [
                self._pool_pair(residual4),
                self._pool_pair(residual8),
                residual4_l1.unsqueeze(1),
                residual4_l2.unsqueeze(1),
                residual4_max.unsqueeze(1),
                residual8_l1.unsqueeze(1),
                residual8_l2.unsqueeze(1),
                residual8_max.unsqueeze(1),
                coarse_l2.unsqueeze(1),
                unexplained_ratio.unsqueeze(1),
                residual_gain.unsqueeze(1),
                detail_concentration.unsqueeze(1),
            ],
            dim=1,
        )
        residual_hidden = self.residual_head(residual_features)
        logits = self.classifier(torch.cat([coarse_features, residual_hidden], dim=1)).view(-1)
        return {
            "logits": logits,
            "coarse_l2": coarse_l2,
            "explained_l2": explained_l2,
            "residual4_l1": residual4_l1,
            "residual4_l2": residual4_l2,
            "residual4_max": residual4_max,
            "residual8_l1": residual8_l1,
            "residual8_l2": residual8_l2,
            "residual8_max": residual8_max,
            "unexplained_ratio": unexplained_ratio,
            "residual_gain": residual_gain,
            "detail_concentration": detail_concentration,
            "residual_alignment": residual_alignment,
            "material_balance": summary["material_balance"],
            "occupancy_count": summary["occupancy_count"],
            "rank_file_imbalance": summary["rank_file_imbalance"],
            "center_pressure": summary["center_pressure"],
        }


def build_coarse_to_fine_board_residual_pyramid_from_config(
    config: dict[str, Any],
) -> CoarseToFineBoardResidualPyramid:
    return CoarseToFineBoardResidualPyramid(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", config.get("num_blocks", 2))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        residual_scale=float(config.get("residual_scale", 1.0)),
    )
