"""DeltaState + SLG Diffusion (p018) — reversible accumulator + sheaf step.

Source: ``ideas/research/primitives/external_17_delta_state_slg_diffusion_fg_tp.md``
(``primitive_delta_state`` + ``primitive_slg_diffusion``).

DeltaState is the canonical stateful reversible index accumulator from
external_17: a state vector ``h ∈ R^d`` exposed via three callable ops
(``forward``, ``apply_delta``, ``inverse_delta``) sharing the embedding
parameter ``W``. The defining "no-rebrand" novelty is the *triple-
interface contract* with a reversible delta path that produces gradients
identical (modulo summation order) to the refreshed forward.

SLG Diffusion adds a single sheaf-Laplacian diffusion step over the
input-determined legal-move graph with low-rank restriction maps. The
practical chess analogue used here is the alignment-pair graph from
:func:`chess_nn_playground.models.primitives.delta_pair_accumulator._alignment_mask`
(same as p014's E(S)), which is rule-derived from the legal board state
and changes with every position. Restriction maps are factorised as
``F_ij = U_{type(i)} V_{type(j)}^T`` so the per-edge cost is small.

For the static-position trainer ``forward(S)`` is exercised; the
``apply_delta`` / ``inverse_delta`` paths are part of the inference
contract documented in ``implementation_notes.md``.
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
    piece_type_and_square,
)
from chess_nn_playground.models.primitives.delta_pair_accumulator import (
    _alignment_mask,
)


class DeltaStateSLGDiffusionHead(DeltaAccumulatorHead):
    """p018 — DeltaState + SLG Diffusion head on the i193 dual-stream trunk."""

    DEFAULT_ABLATIONS: tuple[str, ...] = (
        "none",
        "zero_delta",
        "trunk_only",
        "shuffle_features",
        "disable_gate",
        "zero_state",
        "identity_restriction_maps",  # F_ij = I (collapses SLG to GCN baseline)
        "no_diffusion",               # skip the diffusion step entirely
    )

    def __init__(
        self,
        *,
        stalk_dim: int = 32,
        diffusion_alpha: float = 0.25,
        diffusion_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        if int(stalk_dim) < 1:
            raise ValueError("stalk_dim must be >= 1")
        if float(diffusion_alpha) <= 0.0:
            raise ValueError("diffusion_alpha must be > 0")
        if int(diffusion_steps) < 1:
            raise ValueError("diffusion_steps must be >= 1")
        self._stalk_dim = int(stalk_dim)
        self._diffusion_alpha = float(diffusion_alpha)
        self._diffusion_steps = int(diffusion_steps)
        super().__init__(**kwargs)

    def build_extras(self) -> None:
        stalk = self._stalk_dim
        self.stalk_proj = nn.Linear(self.accumulator_dim, stalk, bias=False)
        # Restriction-map factors: per-piece-type ``U`` and ``V`` matrices.
        # The full per-edge restriction map is ``F_ij = U_i V_j^T`` so the
        # parameter count is ``O(|types| · stalk)``.
        self.restriction_u = nn.Parameter(torch.empty(PIECE_PLANE_COUNT, stalk))
        self.restriction_v = nn.Parameter(torch.empty(PIECE_PLANE_COUNT, stalk))
        nn.init.normal_(self.restriction_u, mean=0.0, std=0.1)
        nn.init.normal_(self.restriction_v, mean=0.0, std=0.1)
        # Final readout pools the diffused stalk back into a single state.
        self.readout = nn.Linear(2 * stalk, stalk)
        self.state_dim = stalk

    def compute_state(
        self,
        features: ActiveFeatures,
        board: torch.Tensor,
        trunk_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = self.accumulator.gather(features)  # (B, K, d)
        stalks = self.stalk_proj(embeddings)            # (B, K, stalk)
        stalks = stalks * features.valid.unsqueeze(-1)

        piece_type, square = piece_type_and_square(features.indices)

        if self.ablation == "no_diffusion":
            diffused = stalks
            edge_count = stalks.new_zeros(stalks.shape[0])
        else:
            valid = features.valid
            valid_pair = valid.unsqueeze(-1) * valid.unsqueeze(-2)
            eye = torch.eye(
                valid.shape[1], device=valid.device, dtype=valid_pair.dtype
            ).unsqueeze(0)
            valid_pair = valid_pair * (1.0 - eye)
            mask = _alignment_mask(square, square).to(dtype=valid_pair.dtype)
            edge_mask = valid_pair * mask  # (B, K, K)

            if self.ablation == "identity_restriction_maps":
                u_table = stalks.new_ones(PIECE_PLANE_COUNT, self._stalk_dim)
                v_table = stalks.new_ones(PIECE_PLANE_COUNT, self._stalk_dim)
            else:
                u_table = self.restriction_u
                v_table = self.restriction_v
            u_i = u_table[piece_type]  # (B, K, stalk)
            v_j = v_table[piece_type]  # (B, K, stalk)

            diffused = stalks
            for _ in range(self._diffusion_steps):
                # Sheaf diffusion: for each edge (i, j), source j contributes
                #   F_ij · stalks_j = u_i ⊙ (v_j · stalks_j) (rank-1 restriction).
                contribution = (v_j * diffused).sum(dim=-1, keepdim=True)  # (B, K, 1)
                # Aggregate neighbour contributions per row using edge_mask.
                neighbour = torch.einsum("bjk,bj->bk", edge_mask, contribution.squeeze(-1))
                # ``neighbour[k]`` is the summed scalar from column j over edges
                # (j, k). The full diffusion step is ``u_i ⊙ neighbour``.
                neighbour_full = u_i * neighbour.unsqueeze(-1)
                degree = edge_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                diffused = diffused + self._diffusion_alpha * (neighbour_full / degree)
                diffused = diffused * features.valid.unsqueeze(-1)
            edge_count = edge_mask.sum(dim=(1, 2))

        # Combine "DeltaState" sum and the diffused stalk pool.
        sum_state = stalks.sum(dim=1)
        diffused_pool = diffused.sum(dim=1)
        readout_input = torch.cat([sum_state, diffused_pool], dim=1)
        state = self.readout(readout_input)

        diagnostics = {
            "dssg_state_norm": state.norm(dim=1),
            "dssg_sum_norm": sum_state.norm(dim=1),
            "dssg_diffused_norm": diffused_pool.norm(dim=1),
            "dssg_edge_count": edge_count.float(),
        }
        return state, diagnostics


def build_delta_state_slg_diffusion_from_config(
    config: dict[str, Any],
) -> DeltaStateSLGDiffusionHead:
    cfg = dict(config)
    kwargs = merge_kwargs(cfg)
    stalk_dim = int(cfg.get("stalk_dim", 32))
    diffusion_alpha = float(cfg.get("diffusion_alpha", 0.25))
    diffusion_steps = int(cfg.get("diffusion_steps", 1))
    return DeltaStateSLGDiffusionHead(
        stalk_dim=stalk_dim,
        diffusion_alpha=diffusion_alpha,
        diffusion_steps=diffusion_steps,
        **kwargs,
    )


__all__ = [
    "DeltaStateSLGDiffusionHead",
    "build_delta_state_slg_diffusion_from_config",
]
