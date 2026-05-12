"""Barrier-Cut Puzzle Network for idea i198.

Working thesis: a true puzzle exists because the defender cannot
maintain a barrier between the attacking force and a valuable target
(king, queen, promotion square, pinned defender, mating square). A
near-puzzle still pressures the barrier, but a defender's barrier
holds — the attack flow does not reach a valuable target.

This bespoke architecture turns that thesis into an explicit
differentiable barrier-cut computation on the 8x8 board. The encoder
predicts three per-square fields:

* ``attack_field A(x) in R_+^(B, 8, 8)`` — attacker pressure mass.
* ``defense_field D(x) in R_+^(B, 8, 8)`` — local defender barrier
  capacity (how much attack flux a square absorbs).
* ``target_field T(x) in R_+^(B, 8, 8)`` — value of a target sitting
  on a square (king, queen, promotion square, pinned defender, ...).

A ``barrier_steps`` iteration of damped 3x3 diffusion then propagates
attack mass across the board, with the defense field acting as a
per-square cut: at each step, the attack potential is reduced
elementwise by ``defense_field`` (clamped at zero) before being
spatially diffused. Squares with a strong defender barrier act as
cuts that absorb flow; squares with a weak barrier let flow leak
through. The reachable target value

    reachable_target_value = sum_{r, f} u_T(r, f) * T(r, f)

measures how much attack actually arrives at a valuable target after
the barrier has done its work. The barrier-defect, or "defense gap"
field

    defense_gap(r, f) = max(0, A_diffuse(r, f) - D(r, f))

tracks where the barrier is locally insufficient. The classifier
reads pooled summaries of these fields plus pooled trunk features to
emit one puzzle logit: high reachable-target value or large defense
gap drives the position toward the puzzle class; a barrier that
absorbs attack mass before it reaches any target drives it toward
non-puzzle.

The architecture is materially distinct from:

* ``ResearchPacketProbe`` — no attack/defense/target fields, no
  barrier diffusion, no min-cut head.
* Sheaf / transport models (i010-i040 family) — those compute global
  Hodge / Sinkhorn statistics; this one runs an explicit iterative
  attack diffusion damped by a learnable defender barrier.
* ``KingEscapePercolationNetwork`` (i007) — that model percolates
  the king out of attacker squares; this one percolates an attacker
  potential field into target squares through a defender barrier.

Tensor contract (``input_channels = 18``):

* input ``x``                 shape ``(B, 18, 8, 8)``
* trunk feats                 shape ``(B, channels, 8, 8)``
* attack/defense/target       shape ``(B, 8, 8)`` each
* barrier potential per step  shape ``(B, 8, 8)``
* reachable target value      shape ``(B,)``
* puzzle ``logits``           shape ``(B,)``
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class _BoardFieldEncoder(nn.Module):
    """Compact convolutional trunk that maps the input board to three
    per-square non-negative fields: attacker mass, defender barrier
    capacity, and target value.
    """

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
        if channels < 1:
            raise ValueError("channels must be >= 1")
        layers: list[nn.Module] = []
        in_ch = input_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            else:
                layers.append(nn.GroupNorm(1, channels))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = channels
        self.body = nn.Sequential(*layers)
        self.field_proj = nn.Conv2d(channels, 3, kernel_size=1)
        self.output_channels = channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.body(x)
        raw_fields = self.field_proj(feats)
        # Softplus keeps the three fields non-negative — they have
        # physical meanings (attacker mass, defender barrier capacity,
        # target value) that must not flip sign.
        attack_field = F.softplus(raw_fields[:, 0])
        defense_field = F.softplus(raw_fields[:, 1])
        target_field = F.softplus(raw_fields[:, 2])
        return feats, attack_field, defense_field, target_field


class _BarrierCutDiffusion(nn.Module):
    """Iterative attack-flow diffusion damped by a defender barrier.

    At each step the attack potential ``u_t`` is locally absorbed by
    the defender field (``relu(u_t - decay_scale * D)``), then
    spatially diffused using a learnable 3x3 conv with non-negative
    weights so that flow can leak around weak barriers but is
    absorbed by strong ones. ``barrier_steps`` controls how many
    iterations of this diffusion happen.
    """

    def __init__(
        self,
        barrier_steps: int = 4,
        decay_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if barrier_steps < 1:
            raise ValueError("barrier_steps must be >= 1")
        if decay_scale <= 0.0:
            raise ValueError("decay_scale must be positive")
        self.barrier_steps = int(barrier_steps)
        self.decay_scale = float(decay_scale)
        # 3x3 diffusion kernel with non-negative entries (softplus on a
        # learned parameter), normalized to a probability simplex so
        # mass is conserved up to the defender absorption term.
        init = torch.full((3, 3), -1.5)
        init[1, 1] = 0.5  # bias the kernel slightly toward staying put.
        self.kernel_logits = nn.Parameter(init)

    def diffusion_kernel(self) -> torch.Tensor:
        weights = F.softmax(self.kernel_logits.view(-1), dim=0).view(1, 1, 3, 3)
        return weights

    def forward(
        self,
        attack_field: torch.Tensor,
        defense_field: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final potential ``u_T`` (B, 8, 8) and the per-step
        absorbed mass ``(B, T)``.
        """
        kernel = self.diffusion_kernel()
        u = attack_field.unsqueeze(1)  # (B, 1, 8, 8)
        defense = defense_field.unsqueeze(1)
        absorbed_per_step: list[torch.Tensor] = []
        for _ in range(self.barrier_steps):
            absorbed = torch.minimum(u, self.decay_scale * defense)
            absorbed_per_step.append(absorbed.squeeze(1).sum(dim=(1, 2)))
            u = (u - absorbed).clamp_min(0.0)
            u = F.conv2d(u, kernel, padding=1)
        absorbed_stack = torch.stack(absorbed_per_step, dim=1)  # (B, T)
        return u.squeeze(1), absorbed_stack


class BarrierCutPuzzleNetwork(nn.Module):
    """Bespoke puzzle_binary classifier built on a barrier-cut
    interpretation of attack vs. defender vs. target.

    Forward returns at least:

    * ``logits``: ``(B,)`` puzzle logit for the BCE-with-logits trainer
      (``(B, num_classes)`` if ``num_classes > 1``).
    * ``prob``: ``sigmoid(logits)`` when ``num_classes == 1``.
    * ``attack_field``, ``defense_field``, ``target_field``:
      ``(B, 8, 8)`` non-negative per-square fields.
    * ``final_attack_potential``: ``(B, 8, 8)`` the diffused attack
      mass after ``barrier_steps`` rounds of barrier absorption.
    * ``reachable_target_value``: ``(B,)`` total attack mass that
      reached a valuable target.
    * ``barrier_absorbed_mass``: ``(B, barrier_steps)`` absorbed mass
      per diffusion step.
    * ``defense_gap``: ``(B, 8, 8)`` ``relu(final_attack - defense)``
      per square.
    * ``defense_gap_mean``, ``defense_gap_max``: ``(B,)`` summary
      scalars of that map.
    * ``barrier_total_absorbed``: ``(B,)`` total mass absorbed by the
      defender barrier across all diffusion steps.
    * ``attack_total_mass``, ``defense_total_capacity``,
      ``target_total_value``: ``(B,)`` global per-field summaries.
    * ``trunk_energy``: ``(B,)`` mean-square trunk activation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        barrier_steps: int = 4,
        decay_scale: float = 1.0,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        height: int = 8,
        width: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.barrier_steps = int(barrier_steps)
        self.decay_scale = float(decay_scale)
        self.dropout_p = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.height = int(height)
        self.width = int(width)

        self.encoder = _BoardFieldEncoder(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            dropout=self.dropout_p,
            use_batchnorm=self.use_batchnorm,
        )
        self.diffusion = _BarrierCutDiffusion(
            barrier_steps=self.barrier_steps,
            decay_scale=self.decay_scale,
        )

        # Head feature pack:
        #   reachable_target_value
        #   attack_total_mass, defense_total_capacity, target_total_value
        #   barrier_total_absorbed
        #   defense_gap_mean, defense_gap_max
        #   final_attack_mean, final_attack_max
        #   trunk_mean, trunk_max
        #   attack_max, defense_max, target_max
        head_in = 14
        self.head_norm = nn.LayerNorm(head_in)
        head_layers: list[nn.Module] = [
            nn.Linear(head_in, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats, attack_field, defense_field, target_field = self.encoder(x)
        final_attack, absorbed_per_step = self.diffusion(attack_field, defense_field)

        reachable_target_value = (final_attack * target_field).sum(dim=(1, 2))
        defense_gap = (final_attack - defense_field).clamp_min(0.0)

        attack_total_mass = attack_field.sum(dim=(1, 2))
        defense_total_capacity = defense_field.sum(dim=(1, 2))
        target_total_value = target_field.sum(dim=(1, 2))
        attack_max = attack_field.amax(dim=(1, 2))
        defense_max = defense_field.amax(dim=(1, 2))
        target_max = target_field.amax(dim=(1, 2))

        barrier_total_absorbed = absorbed_per_step.sum(dim=1)

        defense_gap_mean = defense_gap.mean(dim=(1, 2))
        defense_gap_max = defense_gap.amax(dim=(1, 2))

        final_attack_mean = final_attack.mean(dim=(1, 2))
        final_attack_max = final_attack.amax(dim=(1, 2))

        trunk_mean = feats.mean(dim=(2, 3)).mean(dim=1)
        trunk_max = feats.amax(dim=(2, 3)).mean(dim=1)
        trunk_energy = feats.square().mean(dim=(1, 2, 3))

        head_input = torch.stack(
            [
                reachable_target_value,
                attack_total_mass,
                defense_total_capacity,
                target_total_value,
                barrier_total_absorbed,
                defense_gap_mean,
                defense_gap_max,
                final_attack_mean,
                final_attack_max,
                trunk_mean,
                trunk_max,
                attack_max,
                defense_max,
                target_max,
            ],
            dim=-1,
        )
        head_input = self.head_norm(head_input)
        raw_logits = self.head(head_input)
        logits = _format_logits(raw_logits, self.num_classes)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "attack_field": attack_field,
            "defense_field": defense_field,
            "target_field": target_field,
            "final_attack_potential": final_attack,
            "reachable_target_value": reachable_target_value,
            "barrier_absorbed_mass": absorbed_per_step,
            "barrier_total_absorbed": barrier_total_absorbed,
            "defense_gap": defense_gap,
            "defense_gap_mean": defense_gap_mean,
            "defense_gap_max": defense_gap_max,
            "attack_total_mass": attack_total_mass,
            "defense_total_capacity": defense_total_capacity,
            "target_total_value": target_total_value,
            "trunk_energy": trunk_energy,
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_barrier_cut_puzzle_network_from_config(
    config: dict[str, Any],
) -> BarrierCutPuzzleNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return BarrierCutPuzzleNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        depth=int(cfg.pop("depth", 2)),
        barrier_steps=int(cfg.pop("barrier_steps", 4)),
        decay_scale=float(cfg.pop("decay_scale", 1.0)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        height=int(cfg.pop("height", 8)),
        width=int(cfg.pop("width", 8)),
    )


__all__ = [
    "BarrierCutPuzzleNetwork",
    "build_barrier_cut_puzzle_network_from_config",
]
