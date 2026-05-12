from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


def _norm(channels: int, use_batchnorm: bool) -> nn.Module:
    return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()


class ConvGeluBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            _norm(out_channels, use_batchnorm),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineLogitBranch(nn.Module):
    """Compact simple-CNN-shaped branch that exposes both logit and pooled latent."""

    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        *,
        use_batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        blocks: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(depth):
            blocks.append(
                ConvGeluBlock(
                    in_channels,
                    channels,
                    use_batchnorm=use_batchnorm,
                    dropout=dropout,
                )
            )
            in_channels = channels
        self.features = nn.Sequential(*blocks)
        self.logit_head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.features(x)
        latent = feature_map.mean(dim=(2, 3))
        logit = self.logit_head(latent).view(-1)
        return feature_map, latent, logit


class ResidualAdapterBlock(nn.Module):
    def __init__(self, channels: int, *, use_batchnorm: bool, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=not use_batchnorm),
            _norm(channels, use_batchnorm),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=not use_batchnorm),
            _norm(channels, use_batchnorm),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.body = nn.Sequential(*layers)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class Simple18BoardSummary(nn.Module):
    summary_dim = 26

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
        material_raw = (counts * self.piece_values.to(dtype=x.dtype).view(1, 12)).sum(dim=1)
        side_sign = white_to_move.mul(2.0).sub(1.0)
        material_balance = side_sign * material_raw / 39.0
        material_total = counts[:, [0, 6]].sum(dim=1) + 3.0 * counts[:, [1, 2, 7, 8]].sum(dim=1)
        material_total = material_total + 5.0 * counts[:, [3, 9]].sum(dim=1) + 9.0 * counts[:, [4, 10]].sum(dim=1)
        material_total = material_total / 78.0
        occupancy = piece.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        occupancy_count = occupancy.flatten(1).sum(dim=1) / 32.0
        denom = occupancy.flatten(1).sum(dim=1).clamp_min(1.0)
        rank_imbalance = (occupancy * self.rank_grid.to(dtype=x.dtype)).sum(dim=(1, 2, 3)) / denom
        file_imbalance = (occupancy * self.file_grid.to(dtype=x.dtype)).sum(dim=(1, 2, 3)) / denom
        rank_file_imbalance = torch.sqrt(rank_imbalance.square() + file_imbalance.square())
        center_pressure = (occupancy * self.center_weight.to(dtype=x.dtype)).sum(dim=(1, 2, 3)) / denom
        kings = (piece[:, 5:6] + piece[:, 11:12]).clamp(0.0, 1.0)
        king_ring = F.max_pool2d(kings, kernel_size=3, stride=1, padding=1)
        king_ring_pressure = (occupancy * king_ring).sum(dim=(1, 2, 3)) / denom
        aux = x[:, 13: min(18, x.shape[1])].mean(dim=(2, 3)) if x.shape[1] > 13 else counts.new_zeros(batch, 0)
        if aux.shape[1] < 5:
            aux = F.pad(aux, (0, 5 - aux.shape[1]))
        features = torch.cat(
            [
                counts / 8.0,
                side,
                material_balance.unsqueeze(1),
                material_total.unsqueeze(1),
                occupancy_count.unsqueeze(1),
                rank_imbalance.unsqueeze(1),
                file_imbalance.unsqueeze(1),
                rank_file_imbalance.unsqueeze(1),
                center_pressure.unsqueeze(1),
                king_ring_pressure.unsqueeze(1),
                aux,
            ],
            dim=1,
        )
        return {
            "features": features,
            "material_balance": material_balance,
            "material_total": material_total,
            "occupancy_count": occupancy_count,
            "rank_file_imbalance": rank_file_imbalance,
            "center_pressure": center_pressure,
            "king_ring_pressure": king_ring_pressure,
        }


class FiLMResidualAdapter(nn.Module):
    def __init__(
        self,
        input_channels: int,
        baseline_channels: int,
        baseline_latent_dim: int,
        summary_dim: int,
        adapter_channels: int,
        hidden_dim: int,
        depth: int,
        *,
        use_batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels + 2, adapter_channels, kernel_size=1, bias=not use_batchnorm),
            _norm(adapter_channels, use_batchnorm),
            nn.GELU(),
        )
        self.baseline_projection = nn.Conv2d(baseline_channels, adapter_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                ResidualAdapterBlock(adapter_channels, use_batchnorm=use_batchnorm, dropout=dropout)
                for _ in range(max(1, depth))
            ]
        )
        condition_dim = baseline_latent_dim + summary_dim + 1
        self.film = nn.Linear(condition_dim, adapter_channels * 2)
        self.gate = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(adapter_channels * 2 + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_with_coords: torch.Tensor,
        baseline_map: torch.Tensor,
        baseline_latent: torch.Tensor,
        baseline_logit: torch.Tensor,
        summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = torch.cat([baseline_latent, baseline_logit.unsqueeze(1), summary], dim=1)
        gamma, beta = self.film(condition).chunk(2, dim=1)
        adapter = self.input_projection(x_with_coords) + self.baseline_projection(baseline_map)
        adapter = adapter * (1.0 + 0.25 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1))
        adapter = adapter + beta.unsqueeze(-1).unsqueeze(-1)
        adapter = self.blocks(adapter)
        pooled = torch.cat([adapter.mean(dim=(2, 3)), adapter.amax(dim=(2, 3))], dim=1)
        adapter_features = torch.cat([pooled, condition], dim=1)
        residual_logit = self.residual_head(adapter_features).view(-1)
        residual_gate = torch.sigmoid(self.gate(condition).view(-1))
        return residual_logit, residual_gate, adapter, adapter_features


@dataclass(frozen=True)
class BaselineLogitResidualAdapterConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True
    adapter_channels: int = 0
    residual_scale: float = 1.0
    detach_baseline_context: bool = True


class BaselineLogitResidualAdapter(nn.Module):
    """Puzzle-binary classifier with an explicit baseline logit plus residual adapter."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        adapter_channels: int = 0,
        residual_scale: float = 1.0,
        detach_baseline_context: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("BaselineLogitResidualAdapter supports puzzle_binary single-logit output")
        if channels < 1:
            raise ValueError("channels must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        adapter_width = int(adapter_channels) if int(adapter_channels) > 0 else max(8, channels // 2)
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.summary = Simple18BoardSummary()
        self.baseline = BaselineLogitBranch(
            input_channels=input_channels,
            channels=channels,
            depth=depth,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )
        self.adapter = FiLMResidualAdapter(
            input_channels=input_channels,
            baseline_channels=channels,
            baseline_latent_dim=channels,
            summary_dim=Simple18BoardSummary.summary_dim,
            adapter_channels=adapter_width,
            hidden_dim=hidden_dim,
            depth=depth,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
        )
        rank = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        self.register_buffer("coordinate_planes", torch.cat([rank, file], dim=1), persistent=False)
        self.residual_scale = float(residual_scale)
        self.detach_baseline_context = bool(detach_baseline_context)
        self.config = BaselineLogitResidualAdapterConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            adapter_channels=adapter_width,
            residual_scale=float(residual_scale),
            detach_baseline_context=bool(detach_baseline_context),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        summary = self.summary(x)
        baseline_map, baseline_latent, baseline_logit = self.baseline(x)
        adapter_map = baseline_map.detach() if self.detach_baseline_context else baseline_map
        adapter_latent = baseline_latent.detach() if self.detach_baseline_context else baseline_latent
        adapter_logit = baseline_logit.detach() if self.detach_baseline_context else baseline_logit
        adapter_summary = summary["features"].detach() if self.detach_baseline_context else summary["features"]
        coords = self.coordinate_planes.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1)
        residual_logit, residual_gate, adapter_field, adapter_features = self.adapter(
            torch.cat([x, coords], dim=1),
            adapter_map,
            adapter_latent,
            adapter_logit,
            adapter_summary,
        )
        adapter_correction = self.residual_scale * residual_gate * residual_logit
        logits = baseline_logit + adapter_correction
        return {
            "logits": logits,
            "baseline_logit": baseline_logit,
            "residual_logit": residual_logit,
            "adapter_correction": adapter_correction,
            "residual_gate": residual_gate,
            "baseline_probability": torch.sigmoid(baseline_logit),
            "residual_to_baseline_ratio": adapter_correction.abs() / baseline_logit.abs().clamp_min(1.0e-3),
            "baseline_latent_norm": baseline_latent.norm(dim=1),
            "adapter_feature_norm": adapter_features.norm(dim=1),
            "adapter_field_energy": adapter_field.square().mean(dim=(1, 2, 3)),
            "material_balance": summary["material_balance"],
            "material_total": summary["material_total"],
            "occupancy_count": summary["occupancy_count"],
            "rank_file_imbalance": summary["rank_file_imbalance"],
            "center_pressure": summary["center_pressure"],
            "king_ring_pressure": summary["king_ring_pressure"],
        }


def build_baseline_logit_residual_adapter_from_config(config: dict[str, Any]) -> BaselineLogitResidualAdapter:
    return BaselineLogitResidualAdapter(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", config.get("num_blocks", 2))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        adapter_channels=int(config.get("adapter_channels", 0)),
        residual_scale=float(config.get("residual_scale", 1.0)),
        detach_baseline_context=bool(config.get("detach_baseline_context", True)),
    )
