"""Piece Liability Gradient Network for idea i202.

Working thesis (from
``ideas/i202_piece_liability_gradient_network/math_thesis.md``): in
many puzzles a piece is not merely attacked, it is *liable* -- it
cannot move, defend, capture, or stay without losing something. Near
puzzles may attack pieces but the liability does not propagate through
the position.

This bespoke architecture turns that thesis into an explicit
differentiable pipeline:

1. **Compact convolutional trunk.** ``feats = trunk(x)`` runs ``depth``
   ``Conv2d -> Norm -> GELU -> Dropout2d`` blocks (BatchNorm or
   GroupNorm) and emits ``(B, channels, 8, 8)``.
2. **Piece presence mask.** Planes ``0..11`` of the simple_18 contract
   are the per-piece occupancy bitboards, so
   ``piece_mask[b, s] = clip(sum_{p<12} x[b, p, h, w], 0, 1)`` gives a
   per-square indicator that an actual piece sits on that square.
   Liability lives only on occupied squares.
3. **Action affordance scores.** A ``1x1`` convolution produces an
   ``A``-channel tensor ``affordance[b, a, s] in R`` of action values
   for each affordance type at each square. Default ``num_affordances
   = 4`` follows the thesis: ``move``, ``defend``, ``capture``, ``stay``.
   For an occupied square the higher the score, the better that action
   would be for the piece sitting there.
4. **Initial liability score.** A piece is liable if *every* affordance
   is bad. We therefore take a soft minimum across affordances::

       soft_min[b, s] = -tau * logsumexp(-affordance[b, :, s] / tau)
       L_0[b, s]      = sigmoid(-soft_min[b, s] / lambda) * piece_mask[b, s]

   so ``L_0[b, s] in [0, 1]``, near 1 when even the best affordance is
   bad and the square is occupied, near 0 when at least one affordance
   is good or the square is empty. The thesis "near-puzzles attack but
   liability does not propagate" is preserved: a piece can be attacked
   (poor ``capture`` and ``stay`` scores) yet still have a good
   ``move`` or ``defend`` value.
5. **Liability propagation rounds.** ``propagation_rounds = K``
   iterations propagate liability through learned spatial relation
   kernels. ``relation_count = R`` row-stochastic kernels
   ``relations[r, s, s'] = softmax_{s'} relation_logits[r, s, s']``
   model the "if my defender is liable, so am I" / "if my retreat
   square's defender is liable, so is the retreat" propagators. A
   per-round, per-relation gate ``gate[t, r] in [0, 1]`` controls how
   much liability flows along each relation each round::

       L_propagated[b, r, s] = sum_{s'} relations[r, s, s'] * L_t[b, s']
       delta[b, s]           = sum_r gate[t, r] * L_propagated[b, r, s]
       L_{t+1}[b, s]         = L_t[b, s] + (1 - L_t[b, s]) * delta[b, s] * piece_mask[b, s]

   The probabilistic-OR update keeps every liability score bounded in
   ``[0, 1]`` and never lets liability grow on empty squares. With
   ``relation_count = 1`` and a uniform kernel, propagation collapses
   to the ``no_propagation`` ablation.
6. **Liability gradient and aggregate features.** The final liability
   field ``L_K`` is summarised by ``max_liability``,
   ``mean_liability`` (over occupied squares), ``top_k_liability``
   (mean of the top ``k = liability_top_k`` values), and the
   propagation magnitude ``liability_gradient = mean(L_K - L_0)``. The
   pooled trunk summary ``(mean, max, energy)`` is concatenated to
   give the head input.
7. **Classifier head.** A
   ``LayerNorm -> Linear(hidden_dim) -> GELU -> Dropout -> Linear(num_classes)``
   MLP returns one puzzle logit. A position with one or more pieces
   whose liability is strongly amplified by propagation pushes the
   model toward the puzzle class; positions where attacks are present
   but liability does not propagate stay near the non-puzzle side.

Material distinctness from the shared ``ResearchPacketProbe`` scaffold:

* The probe never builds an action-affordance head, a piece-presence
  mask, or a liability propagation layer.
* The probe never exposes ``action_affordances``,
  ``initial_liability``, ``final_liability``,
  ``liability_trajectory``, or row-stochastic ``relation_kernels`` --
  this network does.
* The ablations called out by the markdown
  (``no_propagation``, ``one_round_only``, ``no_action_decomposition``,
  ``narrow_trunk``) collapse the architecture to weaker baselines via
  the config switches ``relation_count: 1``, ``propagation_rounds:
  1``, ``num_affordances: 1``, ``channels: 32``.

The architecture is strictly board-only: CRTK / source / verification /
engine metadata is reporting-only and never enters the model.

Tensor contract (``input_channels = 18``, ``S = 64`` squares,
``A = num_affordances``, ``R = relation_count``,
``K = propagation_rounds``):

* input ``x``                   shape ``(B, 18, 8, 8)``
* trunk feats                   shape ``(B, channels, 8, 8)``
* piece_mask                    shape ``(B, S)``
* action_affordances            shape ``(B, A, S)``
* initial_liability             shape ``(B, S)``
* relation_kernels              shape ``(R, S, S)``
* propagation_gates             shape ``(K, R)``
* liability_trajectory          shape ``(B, K + 1, S)``
* final_liability               shape ``(B, S)``
* liability_gradient            shape ``(B, S)``
* puzzle ``logits``             shape ``(B,)``
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.idea_blocks import (
    BoardTensorSpec,
    require_board_tensor,
)


AFFORDANCE_TOKEN_NAMES: tuple[str, ...] = (
    "move",
    "defend",
    "capture",
    "stay",
)


def _format_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    return logits.view(-1) if num_classes == 1 else logits


class _BoardTrunk(nn.Module):
    """Compact convolutional trunk over the board planes."""

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
            layers.append(
                nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=not use_batchnorm)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels))
            else:
                layers.append(nn.GroupNorm(1, channels))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = channels
        self.body = nn.Sequential(*layers)
        self.output_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class PieceLiabilityGradientNetwork(nn.Module):
    """Bespoke implementation of idea i202.

    Forward output dict (board-only inputs):

    * ``logits`` ``(B,)`` puzzle logit for the BCE-with-logits trainer
      (``(B, num_classes)`` if ``num_classes > 1``).
    * ``prob`` ``sigmoid(logits)`` when ``num_classes == 1``.
    * ``piece_mask`` ``(B, S)`` per-square piece occupancy in
      ``[0, 1]``.
    * ``action_affordances`` ``(B, A, S)`` raw affordance scores.
    * ``initial_liability`` ``(B, S)`` ``L_0``.
    * ``final_liability`` ``(B, S)`` ``L_K``.
    * ``liability_trajectory`` ``(B, K + 1, S)`` full liability tape.
    * ``liability_gradient`` ``(B, S)`` ``L_K - L_0``.
    * ``relation_kernels`` ``(R, S, S)`` row-stochastic propagation
      kernels.
    * ``propagation_gates`` ``(K, R)`` per-round, per-relation flow
      gates in ``[0, 1]``.
    * ``max_liability`` ``(B,)``.
    * ``mean_liability`` ``(B,)`` masked by piece occupancy.
    * ``top_k_liability`` ``(B,)`` mean of the top ``liability_top_k``
      entries.
    * ``trunk_energy`` ``(B,)`` mean-square trunk activation.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        channels: int = 64,
        hidden_dim: int = 96,
        depth: int = 2,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        num_affordances: int = 4,
        relation_count: int = 8,
        propagation_rounds: int = 4,
        affordance_temperature: float = 1.0,
        liability_scale: float = 1.0,
        liability_top_k: int = 4,
        num_piece_planes: int = 12,
        height: int = 8,
        width: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if num_affordances < 1:
            raise ValueError("num_affordances must be >= 1")
        if relation_count < 1:
            raise ValueError("relation_count must be >= 1")
        if propagation_rounds < 1:
            raise ValueError("propagation_rounds must be >= 1")
        if affordance_temperature <= 0.0:
            raise ValueError("affordance_temperature must be > 0")
        if liability_scale <= 0.0:
            raise ValueError("liability_scale must be > 0")
        if liability_top_k < 1:
            raise ValueError("liability_top_k must be >= 1")
        if num_piece_planes < 1:
            raise ValueError("num_piece_planes must be >= 1")
        if num_piece_planes > input_channels:
            raise ValueError("num_piece_planes cannot exceed input_channels")

        self.spec = BoardTensorSpec(
            input_channels=int(input_channels), height=int(height), width=int(width)
        )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout_p = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)
        self.num_affordances = int(num_affordances)
        self.relation_count = int(relation_count)
        self.propagation_rounds = int(propagation_rounds)
        self.affordance_temperature = float(affordance_temperature)
        self.liability_scale = float(liability_scale)
        self.liability_top_k = int(liability_top_k)
        self.num_piece_planes = int(num_piece_planes)
        self.height = int(height)
        self.width = int(width)
        self.num_squares = self.height * self.width
        if self.liability_top_k > self.num_squares:
            raise ValueError("liability_top_k cannot exceed number of squares")

        self.trunk = _BoardTrunk(
            input_channels=self.input_channels,
            channels=self.channels,
            depth=self.depth,
            dropout=self.dropout_p,
            use_batchnorm=self.use_batchnorm,
        )

        # Per-square action-affordance head: scores how good each
        # candidate action would be for the piece on this square.
        self.affordance_logits = nn.Conv2d(
            self.channels, self.num_affordances, kernel_size=1
        )
        # Spatial relation kernel logits; row-stochastic over the
        # destination square (last axis).
        self.relation_logits = nn.Parameter(
            torch.zeros(self.relation_count, self.num_squares, self.num_squares)
        )
        # Per-round, per-relation propagation gates (sigmoid-bounded).
        self.propagation_gate_logits = nn.Parameter(
            torch.full((self.propagation_rounds, self.relation_count), -1.0)
        )

        # Head feature pack:
        #   [max, mean, top_k, gradient_mean, gradient_max,
        #    propagation_amplification]                    (6)
        #   pooled trunk summary (mean, max, energy)       (3)
        head_in = 6 + 3
        self.head_norm = nn.LayerNorm(head_in)
        head_layers: list[nn.Module] = [
            nn.Linear(head_in, self.hidden_dim),
            nn.GELU(),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.head = nn.Sequential(*head_layers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def affordance_token_names(self) -> tuple[str, ...]:
        names = list(AFFORDANCE_TOKEN_NAMES[: self.num_affordances])
        while len(names) < self.num_affordances:
            names.append(f"affordance_{len(names)}")
        return tuple(names)

    def _piece_mask(self, x: torch.Tensor) -> torch.Tensor:
        piece_planes = x[:, : self.num_piece_planes]
        # Sum across piece planes; clamp to [0, 1] in case the encoder
        # passes overlapping or non-binary planes.
        mask = piece_planes.sum(dim=1).clamp(0.0, 1.0)
        return mask.flatten(1)  # (B, S)

    def _pool_trunk(self, feats: torch.Tensor) -> torch.Tensor:
        mean = feats.mean(dim=(2, 3)).mean(dim=-1)
        max_pool = feats.amax(dim=(2, 3)).mean(dim=-1)
        energy = feats.square().mean(dim=(2, 3)).mean(dim=-1)
        return torch.stack([mean, max_pool, energy], dim=-1)  # (B, 3)

    def _initial_liability(
        self, affordance: torch.Tensor, piece_mask: torch.Tensor
    ) -> torch.Tensor:
        # Soft minimum of action values across affordances.
        tau = self.affordance_temperature
        soft_min = -tau * torch.logsumexp(-affordance / tau, dim=1)  # (B, S)
        # High liability when even the soft-minimum option is bad.
        liability = torch.sigmoid(-soft_min / self.liability_scale)
        return liability * piece_mask

    def _propagation_step(
        self,
        L: torch.Tensor,
        relations: torch.Tensor,
        gate: torch.Tensor,
        piece_mask: torch.Tensor,
    ) -> torch.Tensor:
        # L_propagated[b, r, s] = sum_{s'} relations[r, s, s'] * L[b, s']
        L_propagated = torch.einsum("rsq,bq->brs", relations, L)
        # Mix relations with the per-relation gate.
        delta = (L_propagated * gate.view(1, -1, 1)).sum(dim=1)  # (B, S)
        delta = delta.clamp(0.0, 1.0)
        # Probabilistic-OR update, restricted to occupied squares.
        L_new = L + (1.0 - L) * delta * piece_mask
        return L_new

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        feats = self.trunk(x)
        trunk_summary = self._pool_trunk(feats)  # (B, 3)
        piece_mask = self._piece_mask(x)  # (B, S)

        affordance_map = self.affordance_logits(feats)  # (B, A, 8, 8)
        affordance = affordance_map.flatten(2)  # (B, A, S)

        L_0 = self._initial_liability(affordance, piece_mask)
        relations = self.relation_logits.softmax(dim=-1)  # (R, S, S)
        propagation_gates = torch.sigmoid(self.propagation_gate_logits)  # (K, R)

        L = L_0
        trajectory = [L_0]
        for t in range(self.propagation_rounds):
            L = self._propagation_step(
                L=L,
                relations=relations,
                gate=propagation_gates[t],
                piece_mask=piece_mask,
            )
            trajectory.append(L)

        liability_trajectory = torch.stack(trajectory, dim=1)  # (B, K+1, S)
        L_K = L
        liability_gradient = L_K - L_0  # (B, S)

        max_liability = L_K.amax(dim=-1)
        denom = piece_mask.sum(dim=-1).clamp_min(1.0)
        mean_liability = (L_K * piece_mask).sum(dim=-1) / denom
        top_k = min(self.liability_top_k, self.num_squares)
        top_k_liability = L_K.topk(k=top_k, dim=-1).values.mean(dim=-1)
        gradient_mean = liability_gradient.mean(dim=-1)
        gradient_max = liability_gradient.amax(dim=-1)
        # Amplification: how much further L_K has travelled away from
        # L_0 relative to the initial liability mass.
        propagation_amplification = (
            liability_gradient.abs().mean(dim=-1)
            / (L_0.mean(dim=-1) + 1.0e-3)
        )

        head_input = torch.stack(
            [
                max_liability,
                mean_liability,
                top_k_liability,
                gradient_mean,
                gradient_max,
                propagation_amplification,
            ],
            dim=-1,
        )
        head_input = torch.cat([head_input, trunk_summary], dim=-1)
        head_input = self.head_norm(head_input)
        raw_logits = self.head(head_input)
        logits = _format_logits(raw_logits, self.num_classes)

        output: dict[str, torch.Tensor] = {
            "logits": logits,
            "piece_mask": piece_mask,
            "action_affordances": affordance,
            "initial_liability": L_0,
            "final_liability": L_K,
            "liability_trajectory": liability_trajectory,
            "liability_gradient": liability_gradient,
            "relation_kernels": relations,
            "propagation_gates": propagation_gates,
            "max_liability": max_liability,
            "mean_liability": mean_liability,
            "top_k_liability": top_k_liability,
            "trunk_energy": feats.square().mean(dim=(1, 2, 3)),
        }
        if self.num_classes == 1:
            output["prob"] = torch.sigmoid(logits)
        return output


def build_piece_liability_gradient_network_from_config(
    config: dict[str, Any],
) -> PieceLiabilityGradientNetwork:
    cfg = dict(config)
    cfg.pop("name", None)
    cfg.pop("mechanism_family", None)
    cfg.pop("packet_profile", None)
    return PieceLiabilityGradientNetwork(
        input_channels=int(cfg.pop("input_channels", 18)),
        num_classes=int(cfg.pop("num_classes", 1)),
        channels=int(cfg.pop("channels", 64)),
        hidden_dim=int(cfg.pop("hidden_dim", 96)),
        depth=int(cfg.pop("depth", 2)),
        dropout=float(cfg.pop("dropout", 0.1)),
        use_batchnorm=bool(cfg.pop("use_batchnorm", True)),
        num_affordances=int(cfg.pop("num_affordances", 4)),
        relation_count=int(cfg.pop("relation_count", 8)),
        propagation_rounds=int(cfg.pop("propagation_rounds", 4)),
        affordance_temperature=float(cfg.pop("affordance_temperature", 1.0)),
        liability_scale=float(cfg.pop("liability_scale", 1.0)),
        liability_top_k=int(cfg.pop("liability_top_k", 4)),
        num_piece_planes=int(cfg.pop("num_piece_planes", 12)),
        height=int(cfg.pop("height", 8)),
        width=int(cfg.pop("width", 8)),
    )


__all__ = [
    "AFFORDANCE_TOKEN_NAMES",
    "PieceLiabilityGradientNetwork",
    "build_piece_liability_gradient_network_from_config",
]
