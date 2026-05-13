"""Sparse-Delta Accumulator (p013) — SDA primitive on the i193 trunk.

Source: ``ideas/research/primitives/external_07_sparse_delta_accumulator_segment_scatter.md``
(``primitive_sda``).

The Sparse-Delta Accumulator (SDA) is the cleanest differentiable
generalisation of the HalfKA feature-transformer accumulator: a persistent
state vector ``h ∈ R^d`` is updated by a signed-delta stream
``(I^+_t, I^-_t)`` of feature indices,

    h_t = h_{t-1} + Σ_{i∈I^+_t} W[i] − Σ_{j∈I^-_t} W[j]

with the make/unmake interface ``apply_delta(h, removals, additions)``
exposed alongside ``forward_full(S)``. The make/unmake cost is
``O(|Δ|·d)`` instead of ``O(|S|·d)`` for ``nn.EmbeddingBag`` and the
defining novelty is the *stateful* autograd contract.

At static scout training, every position is a *full forward*: there is no
parent state to carry across mini-batches, so the trainer evaluates the
analytical fixed point ``h = Σ_{(t,s) ∈ S(x)} W[t·64 + s]`` via the shared
:class:`DeltaAccumulator`. The chess-relevant ``O(|Δ|)`` property is
preserved at inference time when the same model wraps a make/unmake
search loop (see ``implementation_notes.md``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives._delta_head_base import (
    DeltaAccumulatorHead,
    merge_kwargs,
)
from chess_nn_playground.models.primitives.delta_accumulator import ActiveFeatures


class SparseDeltaAccumulator(DeltaAccumulatorHead):
    """p013 — Sparse-Delta Accumulator head on the i193 dual-stream trunk."""

    def __init__(self, *, projection_dim: int | None = None, **kwargs: Any) -> None:
        self._projection_dim = projection_dim
        super().__init__(**kwargs)

    def build_extras(self) -> None:
        proj = int(self._projection_dim if self._projection_dim is not None else self.accumulator_dim)
        if proj < 1:
            raise ValueError("projection_dim must be >= 1")
        self.projection = nn.Linear(self.accumulator_dim, proj)
        # ClippedReLU mirrors NNUE's feature-transformer non-linearity. See
        # https://www.chessprogramming.org/NNUE — it is the standard
        # post-accumulator activation and keeps the state bounded for the
        # quantised inference path.
        self.activation = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.state_dim = proj

    def compute_state(
        self,
        features: ActiveFeatures,
        board: torch.Tensor,
        trunk_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.accumulator(features)  # (B, accumulator_dim)
        projected = self.projection(h)
        state = self.activation(projected)
        eps = 1.0e-6
        saturated_low = (projected <= 0.0).float().mean(dim=1)
        saturated_high = (projected >= 1.0).float().mean(dim=1)
        in_range = 1.0 - saturated_low - saturated_high
        diagnostics = {
            "sda_pre_norm": projected.norm(dim=1),
            "sda_post_norm": state.norm(dim=1),
            "sda_saturated_low_frac": saturated_low,
            "sda_saturated_high_frac": saturated_high,
            "sda_in_range_frac": in_range.clamp_min(0.0),
        }
        _ = eps  # keep linter happy; reserved for future numerical guards
        return state, diagnostics


def build_sparse_delta_accumulator_from_config(
    config: dict[str, Any],
) -> SparseDeltaAccumulator:
    cfg = dict(config)
    kwargs = merge_kwargs(cfg)
    projection_dim = cfg.get("projection_dim", None)
    return SparseDeltaAccumulator(
        projection_dim=int(projection_dim) if projection_dim is not None else None,
        **kwargs,
    )


__all__ = [
    "SparseDeltaAccumulator",
    "build_sparse_delta_accumulator_from_config",
]
