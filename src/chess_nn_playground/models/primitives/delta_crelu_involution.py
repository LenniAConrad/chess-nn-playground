"""DeltaCReLU + Involution Reynolds (p015) — saturation-aware delta head.

Source: ``ideas/research/primitives/external_09_delta_crelu_involution_graph_message.md``
(``primitive_delta_crelu`` + ``primitive_involution_reynolds_affine``).

The DeltaCReLU primitive is a stateful affine + ClippedReLU operator
whose forward and backward cost depend on the size of the input edit,
not on the input dimension:

    p_t = p_{t-1} + Σ_k σ_k · E[i_k]
    h_t = clip(p_t, 0, c)

The saturation regime ``s ∈ {below 0, in-range, above c}`` is tracked
per channel so the backward path can mask out gradients for channels
that have crossed a saturation boundary inside the delta window (the
non-trivial piece versus ``F.embedding_bag + F.hardtanh``).

This module also bakes in the second proposal from external_09,
``InvolutionReynoldsAffine``: the colour-swap involution ``ι`` is built
into the operator by augmenting the post-activation state with its
involution-symmetric and -antisymmetric components ``(h + ι h, h − ι h)``.
For the static-position trainer ``ι`` is implemented as the
piece-square involution from
:func:`chess_nn_playground.models.primitives.delta_accumulator.involution_indices`,
which sends ``(piece_type, square)`` to ``(swapped_colour_type,
rank_flipped_square)``. CRTK / source / verification metadata are not
consulted; the involution is rule-derived from the legal board state.
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
    DeltaAccumulator,
    involution_indices,
)


class DeltaCReLUInvolutionHead(DeltaAccumulatorHead):
    """p015 — DeltaCReLU + Involution Reynolds head on the i193 trunk."""

    DEFAULT_ABLATIONS: tuple[str, ...] = (
        "none",
        "zero_delta",
        "trunk_only",
        "shuffle_features",
        "disable_gate",
        "zero_state",
        "no_clip",            # remove the saturation non-linearity (back to additive)
        "no_involution",      # drop the (h ± ι h) involution split
    )

    def __init__(
        self,
        *,
        clip_max: float = 1.0,
        involution_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if float(clip_max) <= 0:
            raise ValueError("clip_max must be > 0")
        self._clip_max = float(clip_max)
        self._involution_weight = float(involution_weight)
        super().__init__(**kwargs)

    def build_extras(self) -> None:
        self.bias = nn.Parameter(torch.zeros(self.accumulator_dim))
        # The state is the concatenation of:
        #   h_clipped, sym = (h + ι h) / 2, asym = (h − ι h) / 2
        # plus a 2-dim saturation summary (low / high saturated fraction).
        self.state_dim = 3 * self.accumulator_dim + 2

    def _involute_features(self, features: ActiveFeatures) -> ActiveFeatures:
        swapped_indices = involution_indices(features.indices)
        return ActiveFeatures(
            indices=swapped_indices,
            valid=features.valid,
            count=features.count,
        )

    def _clip(self, pre: torch.Tensor) -> torch.Tensor:
        if self.ablation == "no_clip":
            return pre
        return pre.clamp(0.0, self._clip_max)

    def compute_state(
        self,
        features: ActiveFeatures,
        board: torch.Tensor,
        trunk_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_raw = self.accumulator(features) + self.bias
        h = self._clip(h_raw)

        if self.ablation == "no_involution":
            sym = torch.zeros_like(h)
            asym = torch.zeros_like(h)
        else:
            invol_features = self._involute_features(features)
            h_swap_raw = self.accumulator(invol_features) + self.bias
            h_swap = self._clip(h_swap_raw)
            sym = 0.5 * (h + h_swap) * self._involution_weight
            asym = 0.5 * (h - h_swap) * self._involution_weight

        low_frac = (h_raw <= 0.0).float().mean(dim=1, keepdim=True)
        high_frac = (h_raw >= self._clip_max).float().mean(dim=1, keepdim=True)
        state = torch.cat([h, sym, asym, low_frac, high_frac], dim=1)

        diagnostics = {
            "dcrelu_pre_norm": h_raw.norm(dim=1),
            "dcrelu_post_norm": h.norm(dim=1),
            "dcrelu_saturated_low_frac": low_frac.squeeze(-1),
            "dcrelu_saturated_high_frac": high_frac.squeeze(-1),
            "dcrelu_involution_sym_norm": sym.norm(dim=1),
            "dcrelu_involution_asym_norm": asym.norm(dim=1),
        }
        return state, diagnostics


def build_delta_crelu_involution_head_from_config(
    config: dict[str, Any],
) -> DeltaCReLUInvolutionHead:
    cfg = dict(config)
    kwargs = merge_kwargs(cfg)
    clip_max = float(cfg.get("clip_max", 1.0))
    involution_weight = float(cfg.get("involution_weight", 1.0))
    return DeltaCReLUInvolutionHead(
        clip_max=clip_max,
        involution_weight=involution_weight,
        **kwargs,
    )


__all__ = [
    "DeltaCReLUInvolutionHead",
    "build_delta_crelu_involution_head_from_config",
]
