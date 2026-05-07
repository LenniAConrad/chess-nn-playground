"""Masked Codec Interaction-Curvature Network for idea i215.

Implements the ``masked codec interaction curvature'' thesis: small
sub-grids of the board are masked at two intensities, the model is
asked to reconstruct them, and the second-difference of reconstruction
error in mask intensity is treated as an interaction-curvature signal.
The architecture is materially distinct from the shared research-packet
probe.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models._packet_bespoke_base import (
    BoardConvStem,
    BoardTensorSpec,
    format_logits,
    require_board_tensor,
)


def _mask_pattern(batch: int, channels: int, height: int, width: int, intensity: float, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> torch.Tensor:
    rng = torch.rand(batch, channels, height, width, device=device, dtype=dtype, generator=generator)
    return (rng < intensity).to(dtype)


class _CodecBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
        )

    def forward(self, masked: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(masked)
        return self.decoder(encoded)


class MaskedCodecInteractionCurvatureNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        intensities: tuple[float, float, float] = (0.05, 0.20, 0.40),
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("MaskedCodecInteractionCurvatureNetwork supports the puzzle_binary one-logit contract")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.num_classes = int(num_classes)
        self.input_channels = int(input_channels)
        if len(intensities) != 3:
            raise ValueError("intensities must contain three increasing values for the curvature finite difference")
        self.intensities = tuple(float(value) for value in intensities)
        self.codec = _CodecBlock(input_channels, channels, dropout)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        head_in = channels + 12
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _reconstruction_error(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        mask = _mask_pattern(x.shape[0], self.input_channels, 8, 8, intensity, x.device, x.dtype)
        masked = x * (1.0 - mask)
        reconstruction = self.codec(masked)
        diff = (reconstruction - x).square()
        return (diff * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        e0 = self._reconstruction_error(x, self.intensities[0])
        e1 = self._reconstruction_error(x, self.intensities[1])
        e2 = self._reconstruction_error(x, self.intensities[2])
        first_diff_low = (e1 - e0) / max(1.0e-6, self.intensities[1] - self.intensities[0])
        first_diff_high = (e2 - e1) / max(1.0e-6, self.intensities[2] - self.intensities[1])
        curvature = first_diff_high - first_diff_low
        feats = self.stem(x)
        feats_pool = feats.mean(dim=(2, 3))
        readout = torch.cat(
            [
                feats_pool,
                e0.unsqueeze(-1),
                e1.unsqueeze(-1),
                e2.unsqueeze(-1),
                first_diff_low.unsqueeze(-1),
                first_diff_high.unsqueeze(-1),
                curvature.unsqueeze(-1),
                (e2 - e0).unsqueeze(-1),
                ((e0 + e1 + e2) / 3.0).unsqueeze(-1),
                (e2 / (e0 + 1.0e-3)).unsqueeze(-1),
                (curvature.abs()).unsqueeze(-1),
                torch.full_like(e0, self.intensities[0]).unsqueeze(-1),
                torch.full_like(e0, self.intensities[2]).unsqueeze(-1),
            ],
            dim=-1,
        )
        readout = self.head_norm(readout)
        logits = format_logits(self.head(readout), self.num_classes)
        return {
            "logits": logits,
            "reconstruction_error_low": e0,
            "reconstruction_error_mid": e1,
            "reconstruction_error_high": e2,
            "interaction_curvature": curvature,
            "interaction_curvature_abs": curvature.abs(),
            "first_difference_low": first_diff_low,
            "first_difference_high": first_diff_high,
            "reconstruction_error_range": e2 - e0,
        }


def build_masked_codec_interaction_curvature_network_from_config(config: dict[str, Any]) -> MaskedCodecInteractionCurvatureNetwork:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    intensities = cfg.get("intensities", (0.05, 0.20, 0.40))
    return MaskedCodecInteractionCurvatureNetwork(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        intensities=tuple(float(value) for value in intensities),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
