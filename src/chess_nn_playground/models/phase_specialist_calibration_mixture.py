"""Phase-Specialist Calibration Mixture for idea i212.

Implements the ``phase-specialist mixture'' thesis: a phase classifier
estimates soft probabilities across opening, middlegame, endgame, and
promotion-race phases; per-phase expert heads produce phase-specialised
logits and per-phase temperature/bias calibration; the mixture is
combined into one puzzle logit. The architecture is materially distinct
from the shared research-packet probe.
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
    us_them_piece_planes,
)


PIECE_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)
PHASE_NAMES = ("opening", "middlegame", "endgame", "promotion_race")


class _PhaseGate(nn.Module):
    def __init__(self, channels: int, num_phases: int) -> None:
        super().__init__()
        self.channels = channels
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, num_phases),
        )

    def forward(self, feats: torch.Tensor, material: torch.Tensor, promotion: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(feats)
        gate_logits = gate_logits + torch.stack(
            [
                32.0 - material,
                (material - 18.0).abs() * -0.2,
                material * -0.5 + 12.0,
                promotion * 6.0,
            ],
            dim=1,
        )
        return F.softmax(gate_logits, dim=-1)


class PhaseSpecialistCalibrationMixture(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        num_phases: int = 4,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("PhaseSpecialistCalibrationMixture supports the puzzle_binary one-logit contract")
        if num_phases != len(PHASE_NAMES):
            raise ValueError(f"num_phases must be {len(PHASE_NAMES)} for this architecture")
        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.num_phases = int(num_phases)
        self.stem = BoardConvStem(input_channels, channels, depth=depth, use_batchnorm=use_batchnorm)
        self.gate = _PhaseGate(channels, num_phases)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_phases)
            ]
        )
        self.calibration = nn.Parameter(torch.zeros(num_phases, 2))
        self.register_buffer("piece_values", torch.tensor(PIECE_VALUES), persistent=False)

    def _material_total(self, x: torch.Tensor) -> torch.Tensor:
        us, them = us_them_piece_planes(x, self.input_channels)
        material = ((us + them) * self.piece_values.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
        return material

    def _promotion_pressure(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels < 12:
            return x.new_zeros(x.shape[0])
        white_pawn_advance = (x[:, 0] * torch.linspace(0.0, 1.0, 8, device=x.device, dtype=x.dtype).view(1, 8, 1)).sum(dim=(1, 2))
        black_pawn_advance = (x[:, 6] * torch.linspace(1.0, 0.0, 8, device=x.device, dtype=x.dtype).view(1, 8, 1)).sum(dim=(1, 2))
        return (white_pawn_advance + black_pawn_advance) / 8.0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.stem(x)
        pooled = torch.cat([feats.mean(dim=(2, 3)), feats.amax(dim=(2, 3))], dim=-1)
        material = self._material_total(x)
        promotion = self._promotion_pressure(x)
        phase_probs = self.gate(feats, material, promotion)
        expert_logits = torch.stack([expert(pooled).squeeze(-1) for expert in self.experts], dim=1)
        scale = F.softplus(self.calibration[:, 0]).view(1, -1)
        bias = self.calibration[:, 1].view(1, -1)
        calibrated = expert_logits * scale + bias
        mixed = (calibrated * phase_probs).sum(dim=1)
        logits = format_logits(mixed.unsqueeze(-1), self.num_classes)
        return {
            "logits": logits,
            "phase_probs": phase_probs,
            "phase_entropy": -(phase_probs.clamp_min(1.0e-6).log() * phase_probs).sum(dim=1),
            "expert_logits": expert_logits,
            "calibrated_expert_logits": calibrated,
            "dominant_phase_index": phase_probs.argmax(dim=1).to(logits.dtype),
            "material_total": material,
            "promotion_pressure": promotion,
        }


def build_phase_specialist_calibration_mixture_from_config(config: dict[str, Any]) -> PhaseSpecialistCalibrationMixture:
    cfg = dict(config)
    cfg.setdefault("num_classes", 1)
    cfg.setdefault("input_channels", 18)
    return PhaseSpecialistCalibrationMixture(
        input_channels=int(cfg["input_channels"]),
        num_classes=int(cfg["num_classes"]),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        num_phases=int(cfg.get("num_phases", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
    )
