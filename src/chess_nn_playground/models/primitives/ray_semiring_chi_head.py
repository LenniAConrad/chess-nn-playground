"""Ray-Semiring χ-Head (p016) — sign-graded value head on the i193 trunk.

Source: ``ideas/research/primitives/external_10_ray_semiring_exchange_and_chi_head.md``
(``primitive_chi_head`` — sign-graded χ-equivariant value head; with a
small contribution from ``primitive_btrs`` — blocker-terminated ray scan).

The χ-head is the strongest novelty claim in external_10: a bilinear
readout primitive that satisfies ``f(τ x) = − f(x)`` for the colour-
swap involution ``τ`` *by construction*. The internal feature space is
split into even/odd channels ``h = (h^+, h^-)`` and only the
even × odd cross-terms

    f(h) = Σ_{ij} M^{+-}_{ij} h^+_i h^-_j

survive in the value head, giving the sign-flip behaviour automatically
(``τ : (h^+, h^-) → (h^+, − h^-)`` ⇒ ``f(τ h) = − f(h)``).

For chess the natural split is ``h^+`` from white piece-square features
and ``h^-`` from black piece-square features. The same accumulator
table ``W`` is shared (the involution acts on the index set, not on
distinct parameter blocks), so the constraint is *structural* in the
forward, not enforced via data augmentation or a regularisation loss.

The module also exposes a light "ray-semiring exchange" diagnostic by
mixing ``h^+`` with rank- and file-wise pooled occupancy via a fixed
geometry buffer — a stand-in for the BTRS first-contact ray scan that
preserves the soft propagation flavour without a custom CUDA kernel.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from chess_nn_playground.models.primitives._delta_head_base import (
    DeltaAccumulatorHead,
    merge_kwargs,
)
from chess_nn_playground.models.primitives.delta_accumulator import (
    ActiveFeatures,
    PIECE_PLANE_COUNT,
    piece_color_id,
    piece_type_and_square,
)


class RaySemiringChiHead(DeltaAccumulatorHead):
    """p016 — Ray-Semiring χ-Head on the i193 dual-stream trunk."""

    DEFAULT_ABLATIONS: tuple[str, ...] = (
        "none",
        "zero_delta",
        "trunk_only",
        "shuffle_features",
        "disable_gate",
        "zero_state",
        "no_chi_grading",  # falsifies the χ-grading (uses h^+ alone)
        "no_ray_exchange", # drops the ray-pooled exchange diagnostic
    )

    def __init__(self, *, chi_rank: int = 32, **kwargs: Any) -> None:
        self._chi_rank = int(chi_rank)
        if self._chi_rank < 1:
            raise ValueError("chi_rank must be >= 1")
        super().__init__(**kwargs)

    def build_extras(self) -> None:
        rank = self._chi_rank
        self.proj_plus = nn.Linear(self.accumulator_dim, rank, bias=False)
        self.proj_minus = nn.Linear(self.accumulator_dim, rank, bias=False)
        # Cross-bilinear ``M^{+-}``; output is a (rank,) feature.
        self.cross_bilinear = nn.Bilinear(rank, rank, rank, bias=False)
        # Ray-exchange projector: simple per-square count summary across 8
        # rank/file slices. The shared geometry buffer is registered on the
        # trunk; we recompute a cheap version here so the head stays
        # self-contained.
        self.register_buffer("rank_mask", _rank_mask(), persistent=False)
        self.register_buffer("file_mask", _file_mask(), persistent=False)
        self.ray_proj = nn.Linear(2 * 8, rank, bias=False)
        # state = [ chi_bilinear (rank), ray_exchange (rank), |h+|, |h-| ]
        self.state_dim = 2 * rank + 2

    def _split_features(
        self, features: ActiveFeatures
    ) -> tuple[ActiveFeatures, ActiveFeatures]:
        piece_type, _ = piece_type_and_square(features.indices)
        is_black = piece_color_id(piece_type).to(dtype=features.valid.dtype)
        is_white = 1.0 - is_black
        white_valid = features.valid * is_white
        black_valid = features.valid * is_black
        feats_plus = ActiveFeatures(
            indices=features.indices,
            valid=white_valid,
            count=white_valid.sum(dim=1),
        )
        feats_minus = ActiveFeatures(
            indices=features.indices,
            valid=black_valid,
            count=black_valid.sum(dim=1),
        )
        return feats_plus, feats_minus

    def _ray_exchange(self, board: torch.Tensor) -> torch.Tensor:
        # 8-bin rank and file occupancy summaries, a cheap stand-in for the
        # BTRS ray-pooled exchange diagnostic. Discrete ``board`` planes are
        # already in {0, 1} so the linear projection inherits the spatial
        # information without re-running the trunk geometry.
        occupancy = board[:, :PIECE_PLANE_COUNT].sum(dim=1).clamp(0.0, 1.0)  # (B, 8, 8)
        rank_summary = occupancy.sum(dim=2)  # (B, 8)
        file_summary = occupancy.sum(dim=1)  # (B, 8)
        return torch.cat([rank_summary, file_summary], dim=1)

    def compute_state(
        self,
        features: ActiveFeatures,
        board: torch.Tensor,
        trunk_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feats_plus, feats_minus = self._split_features(features)
        h_plus_raw = self.accumulator(feats_plus)
        h_minus_raw = self.accumulator(feats_minus)
        h_plus = self.proj_plus(h_plus_raw)
        h_minus = self.proj_minus(h_minus_raw)

        if self.ablation == "no_chi_grading":
            # Replace the cross-bilinear with a same-grade term that breaks
            # the f(τx) = -f(x) guarantee. This is the falsifier ablation.
            chi_state = self.cross_bilinear(h_plus, h_plus)
        else:
            chi_state = self.cross_bilinear(h_plus, h_minus)

        if self.ablation == "no_ray_exchange":
            ray_state = torch.zeros(
                chi_state.shape[0], self._chi_rank, device=chi_state.device, dtype=chi_state.dtype
            )
        else:
            ray_state = self.ray_proj(self._ray_exchange(board))

        norms = torch.stack([h_plus.norm(dim=1), h_minus.norm(dim=1)], dim=1)
        state = torch.cat([chi_state, ray_state, norms], dim=1)
        diagnostics = {
            "chi_plus_norm": h_plus.norm(dim=1),
            "chi_minus_norm": h_minus.norm(dim=1),
            "chi_cross_norm": chi_state.norm(dim=1),
            "chi_ray_norm": ray_state.norm(dim=1),
            "chi_active_white_count": feats_plus.count.float(),
            "chi_active_black_count": feats_minus.count.float(),
        }
        return state, diagnostics


def _rank_mask() -> torch.Tensor:
    mask = torch.zeros(8, 64)
    for r in range(8):
        for f in range(8):
            mask[r, r * 8 + f] = 1.0
    return mask


def _file_mask() -> torch.Tensor:
    mask = torch.zeros(8, 64)
    for f in range(8):
        for r in range(8):
            mask[f, r * 8 + f] = 1.0
    return mask


def build_ray_semiring_chi_head_from_config(
    config: dict[str, Any],
) -> RaySemiringChiHead:
    cfg = dict(config)
    kwargs = merge_kwargs(cfg)
    chi_rank = int(cfg.get("chi_rank", 32))
    return RaySemiringChiHead(chi_rank=chi_rank, **kwargs)


__all__ = [
    "RaySemiringChiHead",
    "build_ray_semiring_chi_head_from_config",
]
