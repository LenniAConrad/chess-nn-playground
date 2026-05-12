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


class FeatureResidualBlock(nn.Module):
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


@dataclass(frozen=True)
class ResidualCalibrationErrorFieldConfig:
    input_channels: int = 18
    num_classes: int = 1
    channels: int = 64
    hidden_dim: int = 96
    depth: int = 2
    dropout: float = 0.1
    use_batchnorm: bool = True
    error_channels: int = 0
    temperature_floor: float = 0.25
    correction_scale: float = 1.0


class ResidualCalibrationErrorField(nn.Module):
    """Baseline CNN with a spatial calibration-error field that rescales logits."""

    field_stats_dim = 6

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        error_channels: int = 0,
        temperature_floor: float = 0.25,
        correction_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("ResidualCalibrationErrorField supports puzzle_binary single-logit output")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channels < 1:
            raise ValueError("channels must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if temperature_floor <= 0:
            raise ValueError("temperature_floor must be positive")
        if correction_scale < 0:
            raise ValueError("correction_scale must be non-negative")

        field_channels = int(error_channels) if int(error_channels) > 0 else max(4, channels // 4)
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.encoder = nn.Sequential(
            ConvNormGelu(input_channels + 2, channels, use_batchnorm=use_batchnorm, dropout=dropout),
            *[FeatureResidualBlock(channels, use_batchnorm=use_batchnorm, dropout=dropout) for _ in range(depth)],
        )
        self.raw_head = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.error_field_head = nn.Sequential(
            ConvNormGelu(channels, channels, use_batchnorm=use_batchnorm, dropout=dropout),
            nn.Conv2d(channels, field_channels, kernel_size=1),
        )
        field_summary_dim = field_channels * 2 + self.field_stats_dim
        self.temperature_head = nn.Sequential(
            nn.Linear(field_summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.correction_head = nn.Sequential(
            nn.Linear(field_summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

        rank = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 8, 1).expand(1, 1, 8, 8)
        file = torch.linspace(-1.0, 1.0, steps=8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        center = (1.0 - torch.maximum(rank.abs(), file.abs())).clamp_min(0.0)
        edge = torch.maximum(rank.abs(), file.abs())
        self.register_buffer("coordinate_planes", torch.cat([rank, file], dim=1), persistent=False)
        self.register_buffer("center_weight", center, persistent=False)
        self.register_buffer("edge_weight", edge, persistent=False)
        self.temperature_floor = float(temperature_floor)
        self.correction_scale = float(correction_scale)
        self.config = ResidualCalibrationErrorFieldConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            error_channels=field_channels,
            temperature_floor=float(temperature_floor),
            correction_scale=float(correction_scale),
        )

    @staticmethod
    def _pool_pair(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)

    def _field_summary(self, error_field: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        abs_field = error_field.abs()
        flat = abs_field.flatten(2)
        mass = flat.sum(dim=2).clamp_min(1.0e-6)
        attention = flat / mass.unsqueeze(2)
        entropy = -(attention * attention.clamp_min(1.0e-8).log()).sum(dim=2).mean(dim=1) / 4.1588830833596715
        energy = torch.sqrt(error_field.flatten(1).square().mean(dim=1).clamp_min(1.0e-12))
        peak = abs_field.flatten(1).amax(dim=1)
        l1 = abs_field.flatten(1).mean(dim=1)
        signed_mean = error_field.flatten(1).mean(dim=1)
        center = self.center_weight.to(dtype=error_field.dtype, device=error_field.device)
        edge = self.edge_weight.to(dtype=error_field.dtype, device=error_field.device)
        total_mass = abs_field.sum(dim=(1, 2, 3)).clamp_min(1.0e-6)
        center_mass = (abs_field * center).sum(dim=(1, 2, 3)) / total_mass
        edge_mass = (abs_field * edge).sum(dim=(1, 2, 3)) / total_mass
        stats = torch.cat(
            [
                energy.unsqueeze(1),
                peak.unsqueeze(1),
                l1.unsqueeze(1),
                entropy.unsqueeze(1),
                center_mass.unsqueeze(1),
                edge_mass.unsqueeze(1),
            ],
            dim=1,
        )
        summary = torch.cat([self._pool_pair(error_field), stats], dim=1)
        return summary, {
            "error_field_energy": energy,
            "error_field_peak": peak,
            "error_field_l1": l1,
            "error_field_entropy": entropy,
            "error_field_center_mass": center_mass,
            "error_field_edge_mass": edge_mass,
            "error_field_signed_mean": signed_mean,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        coords = self.coordinate_planes.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1)
        features = self.encoder(torch.cat([x, coords], dim=1))
        raw_logit = self.raw_head(self._pool_pair(features)).view(-1)
        error_field = self.error_field_head(features)
        field_summary, field_stats = self._field_summary(error_field)
        temperature = F.softplus(self.temperature_head(field_summary).view(-1)) + self.temperature_floor
        correction = self.correction_scale * torch.tanh(self.correction_head(field_summary).view(-1))
        logits = raw_logit / temperature + correction
        raw_probability = torch.sigmoid(raw_logit)
        calibrated_probability = torch.sigmoid(logits)
        confidence_delta = (calibrated_probability - raw_probability).abs()
        calibration_strength = temperature.log().abs() + correction.abs()
        return {
            "logits": logits,
            "raw_logit": raw_logit,
            "calibration_temperature": temperature,
            "calibration_correction": correction,
            "correction_norm": correction.abs(),
            "correction_regularizer": correction.square(),
            "temperature_log": temperature.log(),
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "confidence_delta": confidence_delta,
            "calibration_strength": calibration_strength,
            "error_field": error_field,
            **field_stats,
        }


def build_residual_calibration_error_field_from_config(
    config: dict[str, Any],
) -> ResidualCalibrationErrorField:
    return ResidualCalibrationErrorField(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("depth", config.get("num_blocks", 2))),
        dropout=float(config.get("dropout", 0.1)),
        use_batchnorm=bool(config.get("use_batchnorm", True)),
        error_channels=int(config.get("error_channels", 0)),
        temperature_floor=float(config.get("temperature_floor", 0.25)),
        correction_scale=float(config.get("correction_scale", 1.0)),
    )
