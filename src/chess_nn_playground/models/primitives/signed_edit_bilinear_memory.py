"""Signed-Edit Bilinear Memory (p012) — SEBM primitive on the i193 trunk.

Source: ``ideas/research/primitives/external_01_signed_edit_bilinear_memory_ray_scan.md``
(``primitive_signed_edit_bilinear_memory``).

The Signed-Edit Bilinear Memory primitive maintains a state triple
``(s, u, p) ∈ R^r × R^r × R^r`` over a stream of signed feature edits
``E_t = {(x_j, σ_j)}`` with two learnable per-edit projections
``A x_j → a_j`` and ``B x_j → b_j``:

- Insert (``σ = +1``):  ``p ← p + a_j ⊙ u + b_j ⊙ s``, ``s ← s + a_j``,
  ``u ← u + b_j``.
- Delete (``σ = -1``): the inverse-consistent counterpart.

In a make/unmake search loop the inference cost is ``O(|Δ|·r)``. At
static-position scout training the analytical fixed point of the
recurrence is:

- ``s = Σ_j a_j``  (additive state)
- ``u = Σ_j b_j``  (additive state)
- ``p = s ⊙ u − Σ_j a_j ⊙ b_j``  (pair state — the FM identity for
  sum-of-outer-products restricted to the diagonal direction).

The pair state ``p`` is the load-bearing piece: without it the primitive
collapses to a glorified signed-sum accumulator and is a "rebrand"
(see the failure-mode catalogue in external_01). We therefore expose
``p`` directly to the delta MLP, and the ``ablation = zero_pair_state``
control verifies the pair term is doing work.
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


class SignedEditBilinearMemory(DeltaAccumulatorHead):
    """p012 — Signed-Edit Bilinear Memory head on the i193 dual-stream trunk."""

    DEFAULT_ABLATIONS: tuple[str, ...] = (
        "none",
        "zero_delta",
        "trunk_only",
        "shuffle_features",
        "disable_gate",
        "zero_state",
        "zero_pair_state",   # collapses SEBM to additive-sum baseline
        "diagonal_only",     # keep only Σ a_j ⊙ b_j (drop the s ⊙ u term)
    )

    def __init__(self, *, bilinear_rank: int = 64, **kwargs: Any) -> None:
        self._bilinear_rank = int(bilinear_rank)
        if self._bilinear_rank < 1:
            raise ValueError("bilinear_rank must be >= 1")
        super().__init__(**kwargs)

    def build_extras(self) -> None:
        # ``A`` is the first projection of the active embedding into the
        # bilinear-memory rank ``r``; ``B`` is the second. They share the
        # source embedding table (``self.accumulator.embedding``) so that
        # SEBM remains a *pair* generalisation of the additive accumulator
        # rather than a fully independent representation.
        rank = self._bilinear_rank
        self.proj_a = nn.Linear(self.accumulator_dim, rank, bias=False)
        self.proj_b = nn.Linear(self.accumulator_dim, rank, bias=False)
        # Output dimension is ``[s | u | p] = 3 r``.
        self.state_dim = 3 * rank

    def compute_state(
        self,
        features: ActiveFeatures,
        board: torch.Tensor,
        trunk_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = self.accumulator.gather(features)  # (B, K, accumulator_dim)
        a = self.proj_a(embeddings)  # (B, K, rank)
        b = self.proj_b(embeddings)  # (B, K, rank)
        valid = features.valid.unsqueeze(-1)
        a = a * valid
        b = b * valid

        s = a.sum(dim=1)  # (B, rank)
        u = b.sum(dim=1)  # (B, rank)
        diagonal = (a * b).sum(dim=1)  # Σ_j a_j ⊙ b_j

        if self.ablation == "diagonal_only":
            # Drops the s ⊙ u cross-term, leaving only the diagonal.
            p = diagonal
        elif self.ablation == "zero_pair_state":
            # Pair state ablated; this should match a pure additive baseline.
            p = torch.zeros_like(diagonal)
        else:
            # The pair-state algebraic identity from external_01.
            p = s * u - diagonal

        state = torch.cat([s, u, p], dim=1)
        eps = 1.0e-6
        cross_norm = (s * u).norm(dim=1)
        diag_norm = diagonal.norm(dim=1).clamp_min(eps)
        diagnostics = {
            "sebm_s_norm": s.norm(dim=1),
            "sebm_u_norm": u.norm(dim=1),
            "sebm_p_norm": p.norm(dim=1),
            "sebm_diagonal_norm": diagonal.norm(dim=1),
            "sebm_pair_ratio": cross_norm / diag_norm,
        }
        return state, diagnostics


def build_signed_edit_bilinear_memory_from_config(
    config: dict[str, Any],
) -> SignedEditBilinearMemory:
    cfg = dict(config)
    kwargs = merge_kwargs(cfg)
    bilinear_rank = int(cfg.get("bilinear_rank", kwargs.get("accumulator_dim", 64)))
    return SignedEditBilinearMemory(bilinear_rank=bilinear_rank, **kwargs)


__all__ = [
    "SignedEditBilinearMemory",
    "build_signed_edit_bilinear_memory_from_config",
]
