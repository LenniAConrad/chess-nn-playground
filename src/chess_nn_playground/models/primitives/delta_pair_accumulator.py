"""Δ-Pair Accumulator (p014) — DPA primitive on the i193 trunk.

Source: ``ideas/research/primitives/external_08_delta_pair_ray_selective_bispectrum.md``
(``primitive_delta_pair_accumulator``).

DPA generalises HalfKA's first-order accumulator with an explicit pair
term restricted to an *input-dependent* edge set ``E(S) ⊂ S × S``:

    A(S) = Σ_{i ∈ S} u_i + Σ_{(i,j) ∈ E(S)} W_{type(i), type(j), Δsq(i,j)}

The structural distinction over Rendle's factorisation-machine identity
is that ``E(S)`` is a *strict subset* of pairs, not all of S × S; the
diagonal-subtraction trick therefore cannot recover the pair term and
the pair structure must be enumerated explicitly.

Chess-specific deterministic edge set used here:

    E(S) = { (i, j) ∈ S × S : i ≠ j and pieces at i, j share a
             rank, file, or diagonal (potential alignment) }

This captures every battery / pin / skewer / x-ray candidate without
needing blocker-clearance checks (which would also be a fine
deterministic function of S; alignment is the cheapest cousin and
matches the published spec's "attacker → defender pair" intent). Pair
edges are gathered into pair-type indices ``(piece_type_i,
piece_type_j) -> [0, 144)`` and looked up in a learnable pair embedding
table; the pair contributions are summed and concatenated with the
first-order state ``u``.
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
    piece_type_and_square,
)


PIECE_PAIR_COUNT = PIECE_PLANE_COUNT * PIECE_PLANE_COUNT  # 144


def _alignment_mask(squares_i: torch.Tensor, squares_j: torch.Tensor) -> torch.Tensor:
    """Return a (B, K, K) boolean mask of square pairs sharing a chess line.

    Two squares share a chess line iff they are on the same rank, file, or
    diagonal — the union of bishop and rook attack patterns ignoring
    blockers. The diagonal predicate also handles knight-jumps as a
    special case (rank-diff and file-diff both in {1, 2}, summing to 3).
    """

    ranks_i = (squares_i // 8).unsqueeze(-1)  # (B, K, 1)
    files_i = (squares_i % 8).unsqueeze(-1)
    ranks_j = (squares_j // 8).unsqueeze(-2)  # (B, 1, K)
    files_j = (squares_j % 8).unsqueeze(-2)
    same_rank = ranks_i == ranks_j
    same_file = files_i == files_j
    diag = (ranks_i - ranks_j).abs() == (files_i - files_j).abs()
    return same_rank | same_file | diag


class DeltaPairAccumulator(DeltaAccumulatorHead):
    """p014 — Δ-Pair Accumulator head on the i193 trunk."""

    DEFAULT_ABLATIONS: tuple[str, ...] = (
        "none",
        "zero_delta",
        "trunk_only",
        "shuffle_features",
        "disable_gate",
        "zero_state",
        "zero_pair_term",      # collapse DPA to first-order accumulator (rebrand)
        "uniform_edge_mask",   # use full S × S (no alignment selectivity)
    )

    def __init__(self, *, pair_dim: int = 32, **kwargs: Any) -> None:
        self._pair_dim = int(pair_dim)
        if self._pair_dim < 1:
            raise ValueError("pair_dim must be >= 1")
        super().__init__(**kwargs)

    def build_extras(self) -> None:
        # First-order projection of the active-set sum.
        self.first_order_proj = nn.Linear(self.accumulator_dim, self.accumulator_dim)
        # Pair table indexed by (type_i, type_j). The pair embedding is the
        # ``W_{type(i), type(j), Δsq(i,j)}`` table from external_08 with the
        # Δsq axis approximated by a low-rank conditioning on (rank_diff,
        # file_diff) at scoring time.
        self.pair_table = nn.Embedding(PIECE_PAIR_COUNT, self._pair_dim)
        nn.init.normal_(self.pair_table.weight, mean=0.0, std=0.05)
        # Low-rank Δsq gate (rank_diff, file_diff) → scalar.
        self.delta_square_gate = nn.Sequential(
            nn.Linear(2, self._pair_dim),
            nn.GELU(),
            nn.Linear(self._pair_dim, self._pair_dim),
        )
        self.state_dim = self.accumulator_dim + self._pair_dim

    def compute_state(
        self,
        features: ActiveFeatures,
        board: torch.Tensor,
        trunk_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = self.accumulator.gather(features)  # (B, K, d)
        first_order = self.first_order_proj(embeddings.sum(dim=1))

        piece_type, square = piece_type_and_square(features.indices)
        valid = features.valid  # (B, K)

        # Pairwise validity: both endpoints must be active and i ≠ j.
        valid_pair = valid.unsqueeze(-1) * valid.unsqueeze(-2)  # (B, K, K)
        eye = torch.eye(
            valid.shape[1], device=valid.device, dtype=valid_pair.dtype
        ).unsqueeze(0)
        valid_pair = valid_pair * (1.0 - eye)

        if self.ablation == "uniform_edge_mask":
            edge_mask = valid_pair
        else:
            mask = _alignment_mask(square, square).to(dtype=valid_pair.dtype)
            edge_mask = valid_pair * mask  # (B, K, K)

        type_pair = piece_type.unsqueeze(-1) * PIECE_PLANE_COUNT + piece_type.unsqueeze(-2)
        # Clamp invalid slot indices to a benign value before the lookup; the
        # mask zero-multiplies their contribution out below.
        type_pair_safe = type_pair.clamp(0, PIECE_PAIR_COUNT - 1)
        pair_embed = self.pair_table(type_pair_safe)  # (B, K, K, pair_dim)

        rank_diff = ((square.unsqueeze(-1) // 8) - (square.unsqueeze(-2) // 8)).float() / 8.0
        file_diff = ((square.unsqueeze(-1) % 8) - (square.unsqueeze(-2) % 8)).float() / 8.0
        delta_sq = torch.stack([rank_diff, file_diff], dim=-1)
        # Apply learned Δsq conditioning per pair.
        pair_term = pair_embed * self.delta_square_gate(delta_sq)
        pair_term = pair_term * edge_mask.unsqueeze(-1)

        pair_state = pair_term.sum(dim=(1, 2))

        if self.ablation == "zero_pair_term":
            pair_state = torch.zeros_like(pair_state)

        edge_count = edge_mask.sum(dim=(1, 2))
        full_count = valid_pair.sum(dim=(1, 2)).clamp_min(1.0)
        selectivity = (edge_count / full_count).clamp(0.0, 1.0)
        diagnostics = {
            "dpa_first_order_norm": first_order.norm(dim=1),
            "dpa_pair_state_norm": pair_state.norm(dim=1),
            "dpa_edge_count": edge_count.float(),
            "dpa_edge_selectivity": selectivity,
        }
        state = torch.cat([first_order, pair_state], dim=1)
        return state, diagnostics


def build_delta_pair_accumulator_from_config(
    config: dict[str, Any],
) -> DeltaPairAccumulator:
    cfg = dict(config)
    kwargs = merge_kwargs(cfg)
    pair_dim = int(cfg.get("pair_dim", 32))
    return DeltaPairAccumulator(pair_dim=pair_dim, **kwargs)


__all__ = [
    "DeltaPairAccumulator",
    "build_delta_pair_accumulator_from_config",
]
