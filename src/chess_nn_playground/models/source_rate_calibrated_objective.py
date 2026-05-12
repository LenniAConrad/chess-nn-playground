"""Source-Rate Calibrated Objective network for idea i176.

The markdown thesis (``ideas/all_ideas/registry/i176_source_rate_calibrated_objective``) frames the
puzzle_binary task as a *source-rate* problem: at a target puzzle recall on
fine label 2, how many fine-label-1 near-puzzles are falsely called puzzles?
The proposal adds a differentiable rate-calibrated penalty on top of the
standard BCE, so the trainable surface includes:

- ``logit`` (the puzzle decision score), and
- a soft indicator ``sigmoid((logit - tau) / temp)`` whose mean over fine=1
  estimates the near-puzzle false-positive rate and whose mean over fine=2
  estimates the puzzle recall.

To honour that thesis with bespoke architecture rather than just a loss-side
hack, this module emits the calibration parameters (``tau`` and ``temp``) as
*model state* and constructs the puzzle logit from three explicitly named
evidence channels - puzzle, near-puzzle, and random-negative - so the rate
penalty can be wired straight into the model output without re-implementing
the soft rates downstream.

This makes the model materially distinct from a plain CNN baseline:

- it has a learned decision threshold and temperature,
- the puzzle logit is a calibrated combination of three evidence heads, and
- the forward pass returns the soft indicator the rate-calibrated objective
  consumes per minibatch.

The architecture is intentionally board-only; CRTK / source metadata is
reporting-only and never enters the forward pass.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.idea_blocks import BoardTensorSpec, require_board_tensor


class _Trunk(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: int,
        depth: int,
        dropout: float,
        use_batchnorm: bool,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(depth):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    bias=not use_batchnorm,
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = channels
        self.trunk = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class _EvidenceHead(nn.Module):
    """Per-class evidence MLP over pooled trunk features."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SourceRateCalibratedObjectiveNetwork(nn.Module):
    """Bespoke implementation of idea i176.

    The model produces three evidence channels - puzzle, near-puzzle and
    random-negative - and combines them into a single calibrated puzzle logit.
    Two learnable scalars ``tau`` and ``temp`` parameterise the soft-rate
    sigmoid that the source-rate calibrated loss consumes:

        soft_indicator = sigmoid((logit - tau) / temp)

    Fine-label gated mean of ``soft_indicator`` is the differentiable proxy
    for near-puzzle FP rate (fine=1) and puzzle recall (fine=2) used by the
    rate penalty. The trainer can read the indicator straight off the output
    dict; nothing else is needed to wire the calibrated objective.
    """

    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        near_puzzle_weight_init: float = 1.0,
        random_negative_weight_init: float = 0.25,
        tau_init: float = 0.0,
        temperature_init: float = 0.5,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError(
                "SourceRateCalibratedObjectiveNetwork supports the puzzle_binary one-logit contract"
            )
        if temperature_init <= 0.0:
            raise ValueError("temperature_init must be > 0")

        self.spec = BoardTensorSpec(input_channels=input_channels)
        self.trunk = _Trunk(input_channels, channels, depth, dropout, use_batchnorm)

        pooled_dim = 2 * channels  # mean and max pool concatenated.
        self.puzzle_head = _EvidenceHead(pooled_dim, hidden_dim, dropout)
        self.near_puzzle_head = _EvidenceHead(pooled_dim, hidden_dim, dropout)
        self.random_negative_head = _EvidenceHead(pooled_dim, hidden_dim, dropout)

        # Learnable, strictly positive weights on the negative-evidence channels
        # via softplus(raw) so the calibrated logit always *subtracts*
        # near-puzzle and random-negative evidence from the puzzle evidence.
        self.near_weight_raw = nn.Parameter(
            torch.tensor(_inv_softplus(near_puzzle_weight_init), dtype=torch.float32)
        )
        self.random_weight_raw = nn.Parameter(
            torch.tensor(_inv_softplus(random_negative_weight_init), dtype=torch.float32)
        )

        # Decision threshold tau is a learnable scalar; the soft-rate sigmoid
        # uses it directly. Temperature is parameterised through softplus to
        # keep it positive and finite.
        self.tau = nn.Parameter(torch.tensor(float(tau_init), dtype=torch.float32))
        self.temperature_raw = nn.Parameter(
            torch.tensor(_inv_softplus(temperature_init), dtype=torch.float32)
        )

    @staticmethod
    def _pool(feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([feat.mean(dim=(2, 3)), feat.amax(dim=(2, 3))], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feat = self.trunk(x)
        pooled = self._pool(feat)

        puzzle_evidence = self.puzzle_head(pooled)
        near_evidence = self.near_puzzle_head(pooled)
        random_evidence = self.random_negative_head(pooled)

        near_weight = F.softplus(self.near_weight_raw)
        random_weight = F.softplus(self.random_weight_raw)
        # Calibrated puzzle logit: subtracting weighted negative-evidence
        # channels mirrors the rate-calibrated objective which directly
        # penalises near-puzzle false positives.
        logits = (
            puzzle_evidence
            - near_weight * near_evidence
            - random_weight * random_evidence
        )

        temperature = F.softplus(self.temperature_raw).clamp_min(1.0e-4)
        soft_indicator = torch.sigmoid((logits - self.tau) / temperature)

        bsz = logits.shape[0]
        tau_vec = self.tau.view(1).expand(bsz)
        temperature_vec = temperature.view(1).expand(bsz)
        near_weight_vec = near_weight.view(1).expand(bsz)
        random_weight_vec = random_weight.view(1).expand(bsz)

        return {
            "logits": logits,
            "source_rate_puzzle_evidence": puzzle_evidence,
            "source_rate_near_puzzle_evidence": near_evidence,
            "source_rate_random_negative_evidence": random_evidence,
            "source_rate_threshold_tau": tau_vec,
            "source_rate_temperature": temperature_vec,
            "source_rate_near_evidence_weight": near_weight_vec,
            "source_rate_random_evidence_weight": random_weight_vec,
            "source_rate_soft_indicator": soft_indicator,
        }


def _inv_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("value must be > 0 to invert softplus")
    # softplus(x) = log(1 + exp(x)); invert via log(exp(value) - 1).
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def build_source_rate_calibrated_objective_from_config(
    config: dict[str, Any],
) -> SourceRateCalibratedObjectiveNetwork:
    cfg = dict(config)
    return SourceRateCalibratedObjectiveNetwork(
        input_channels=int(cfg.get("input_channels", 18)),
        channels=int(cfg.get("channels", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        depth=int(cfg.get("depth", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_batchnorm=bool(cfg.get("use_batchnorm", True)),
        near_puzzle_weight_init=float(cfg.get("near_puzzle_weight_init", 1.0)),
        random_negative_weight_init=float(cfg.get("random_negative_weight_init", 0.25)),
        tau_init=float(cfg.get("tau_init", 0.0)),
        temperature_init=float(cfg.get("temperature_init", 0.5)),
        num_classes=int(cfg.get("num_classes", 1)),
    )
