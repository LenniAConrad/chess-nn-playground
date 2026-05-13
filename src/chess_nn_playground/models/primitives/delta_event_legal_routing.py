"""Delta-Event Legal-Move Routing (p017) — selective accumulator head.

Source: ``ideas/research/primitives/external_11_delta_event_legal_move_routing.md``
(``primitive_delta_event_accumulator`` + ``primitive_legal_move_routing``).

The Delta-Event Accumulator forms the additive accumulator state

    A_t = A_{t-1} + Σ_r s_r · Θ[e_r]

over a stream of signed event ids ``e_r`` with signs ``s_r``.
The Legal-Move Routing proposal augments this with an *input-induced
routing weight*: rather than gathering raw event embeddings, each active
piece contributes ``α_i(S) · Θ[e_i]`` where ``α_i(S)`` is computed inside
the operator from chess-rule connectivity (legal moves the piece would
generate). This pushes the rule-derived sparsity into the operator
boundary instead of relying on a pre-computed dense attention mask.

For this static-position implementation, ``α_i(S)`` is derived from the
i193 trunk's deterministic attack-relation tensor (via a board-derived
mobility score: the number of pseudo-legal target squares each active
piece can reach, ignoring blockers). This is a deterministic function of
``S`` and contains no CRTK / source / verification metadata — exactly the
"chess-legality connectivity" the spec asks for. CRTK metadata, source
labels, verification flags, and engine scores are **not** consumed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives._delta_head_base import (
    DeltaAccumulatorHead,
    merge_kwargs,
)
from chess_nn_playground.models.primitives.delta_accumulator import (
    ActiveFeatures,
    PIECE_PLANE_COUNT,
    SQUARES,
    piece_color_id,
    piece_type_and_square,
)


def _build_pseudo_attack_table() -> torch.Tensor:
    """(12, 64, 64) pseudo-attack table for routing.

    Each entry ``table[piece_type, source, target]`` is 1 iff a piece of
    that type on ``source`` can reach ``target`` ignoring blockers. The
    table mirrors the standard chess movement rules so the routing weight
    is a rule-derived count of pseudo-legal target squares.
    """

    table = torch.zeros(PIECE_PLANE_COUNT, SQUARES, SQUARES)
    for source in range(SQUARES):
        sr, sf = divmod(source, 8)
        # Knight (planes 1 and 7).
        for dr, df in (
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1),
        ):
            tr, tf = sr + dr, sf + df
            if 0 <= tr < 8 and 0 <= tf < 8:
                table[1, source, tr * 8 + tf] = 1.0
                table[7, source, tr * 8 + tf] = 1.0
        # King (planes 5 and 11).
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                tr, tf = sr + dr, sf + df
                if 0 <= tr < 8 and 0 <= tf < 8:
                    table[5, source, tr * 8 + tf] = 1.0
                    table[11, source, tr * 8 + tf] = 1.0
        # Pawn (planes 0 white moves up the board towards rank 8, 6 black
        # moves down). simple_18 stores rank 8 at row index 0, rank 1 at
        # row index 7, so white "forward" decrements the row index.
        for color, plane in ((0, 0), (1, 6)):
            forward = -1 if color == 0 else 1
            for df in (-1, 1):
                tr, tf = sr + forward, sf + df
                if 0 <= tr < 8 and 0 <= tf < 8:
                    table[plane, source, tr * 8 + tf] = 1.0
        # Sliders (bishop / rook / queen) — ignore blockers per spec.
        for plane, directions in (
            (2, [(-1, -1), (-1, 1), (1, -1), (1, 1)]),                           # white bishop
            (8, [(-1, -1), (-1, 1), (1, -1), (1, 1)]),                           # black bishop
            (3, [(-1, 0), (1, 0), (0, -1), (0, 1)]),                              # white rook
            (9, [(-1, 0), (1, 0), (0, -1), (0, 1)]),                              # black rook
            (4, [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]),  # white queen
            (10, [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]), # black queen
        ):
            for dr, df in directions:
                tr, tf = sr + dr, sf + df
                while 0 <= tr < 8 and 0 <= tf < 8:
                    table[plane, source, tr * 8 + tf] = 1.0
                    tr += dr
                    tf += df
    return table


class DeltaEventLegalRoutingHead(DeltaAccumulatorHead):
    """p017 — Delta-Event Legal-Move Routing on the i193 dual-stream trunk."""

    DEFAULT_ABLATIONS: tuple[str, ...] = (
        "none",
        "zero_delta",
        "trunk_only",
        "shuffle_features",
        "disable_gate",
        "zero_state",
        "uniform_routing",   # set every α_i = 1 (no routing)
        "scrambled_routing", # shuffle α_i across active pieces
    )

    def __init__(self, *, routing_hidden_dim: int = 32, **kwargs: Any) -> None:
        self._routing_hidden_dim = int(routing_hidden_dim)
        if self._routing_hidden_dim < 1:
            raise ValueError("routing_hidden_dim must be >= 1")
        super().__init__(**kwargs)

    def build_extras(self) -> None:
        self.register_buffer(
            "pseudo_attack_table",
            _build_pseudo_attack_table(),
            persistent=False,
        )
        rh = self._routing_hidden_dim
        self.routing_mlp = nn.Sequential(
            nn.Linear(3, rh),
            nn.GELU(),
            nn.Linear(rh, 1),
        )
        self.state_dim = self.accumulator_dim + 2

    def _routing_weights(
        self, board: torch.Tensor, features: ActiveFeatures
    ) -> torch.Tensor:
        piece_type, square = piece_type_and_square(features.indices)
        # Gather pseudo-legal mobility per active piece (sum over target squares).
        attack_table = self.pseudo_attack_table.to(
            device=board.device, dtype=board.dtype
        )  # (12, 64, 64) — [piece_type, source, target]
        piece_attacks = attack_table[piece_type, square]          # (B, K, 64)
        mobility = piece_attacks.sum(dim=-1)                      # (B, K)
        mobility = mobility * features.valid

        # Side-to-move flag: is this piece our own piece?
        stm = board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)
        is_white = (stm > 0.5).float().unsqueeze(-1)
        is_own_color = (piece_color_id(piece_type) == 0).float()
        own_match = torch.where(
            is_white > 0.5, is_own_color, 1.0 - is_own_color
        ) * features.valid

        routing_inputs = torch.stack(
            [mobility / 28.0, own_match, features.valid], dim=-1
        )  # (B, K, 3)
        weights = torch.sigmoid(self.routing_mlp(routing_inputs).squeeze(-1))
        weights = weights * features.valid
        return weights

    def compute_state(
        self,
        features: ActiveFeatures,
        board: torch.Tensor,
        trunk_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = self.accumulator.gather(features)
        weights = self._routing_weights(board, features)

        if self.ablation == "uniform_routing":
            weights = features.valid
        elif self.ablation == "scrambled_routing":
            perm = torch.randperm(weights.shape[1], device=weights.device)
            weights = weights[:, perm] * features.valid

        weighted = embeddings * weights.unsqueeze(-1)
        h = weighted.sum(dim=1)

        eps = 1.0e-6
        weight_sum = weights.sum(dim=1)
        weight_mean = weight_sum / features.count.clamp_min(1.0)
        # Entropy of weights normalised within each sample.
        normed = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
        entropy = -(normed.clamp_min(eps) * normed.clamp_min(eps).log()).sum(dim=1)
        entropy = entropy / torch.log(features.count.clamp_min(2.0))

        state = torch.cat([h, weight_mean.unsqueeze(-1), weight_sum.unsqueeze(-1)], dim=1)
        diagnostics = {
            "delr_state_norm": h.norm(dim=1),
            "delr_routing_mean": weight_mean,
            "delr_routing_sum": weight_sum,
            "delr_routing_entropy": entropy,
        }
        return state, diagnostics


def build_delta_event_legal_routing_from_config(
    config: dict[str, Any],
) -> DeltaEventLegalRoutingHead:
    cfg = dict(config)
    kwargs = merge_kwargs(cfg)
    routing_hidden_dim = int(cfg.get("routing_hidden_dim", 32))
    return DeltaEventLegalRoutingHead(
        routing_hidden_dim=routing_hidden_dim,
        **kwargs,
    )


__all__ = [
    "DeltaEventLegalRoutingHead",
    "build_delta_event_legal_routing_from_config",
]
