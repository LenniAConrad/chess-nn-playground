"""Incremental Delta-Linear Accumulator Head (p025).

Promotes the **Incremental Delta-Linear Operator (IDL)** primitive from
``ideas/research/primitives/external_21_incremental_delta_linear_color_involution_adjacency.md``.
The research file lists five proposals; per the implementation handoff we
implement the **first-ranked** one (IDL) and document the rest as deferred.

IDL is the differentiable lift of the NNUE accumulator structure:

    S = sum over (piece-type t, square s in 0..63 with x_{t, s} == 1) of W_{t, s}

The output is a low-dim accumulator state ``S`` that depends only on which
piece occupies which square. The sparse-update interpretation (forward in
``O(k)`` time when ``k`` squares change) is preserved by the math even though
our trainer evaluates positions independently — the operator's
**factorisation** is the primitive (per-(piece, square) embedding -> sparse
sum), not the optimisation around incrementalism.

The head is an *additive, gated* side head on the i193 dual-stream trunk
(matching the i248 / i246 contract):

    final_logit = base_logit + sigmoid(MLP_gate(trunk_diagnostics, ||S||))
                                * MLP_delta(S, trunk_diagnostics)

Inputs are the ``simple_18`` board tensor only. CRTK metadata, source labels,
verification flags, engine evaluations, and any other report-only metadata
are **never** consumed by the model.

Deferred IDL-file proposals (kept research-only, not implemented here):
``IEL`` color-involution (a weight-tying constraint that is best built into
the trunk rather than added as a side head), ``AGR`` legal-move adjacency
reduction (covered by ``p027``), ``PPI`` piece-pair interaction kernel
(covered separately by the DHPE family), and ``GSL`` king-conditioned
spatial lookup (already a king-stream feature in the i193 trunk).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.primitives.primitive_heads import (
    SHARED_ABLATIONS,
    BoardTensorSpec,
    build_trunk_from_kwargs,
    extract_trunk_diagnostics,
    fuse_with_base_logit,
    require_board_tensor,
    small_mlp,
    standard_diagnostics_dict,
)


PIECE_PLANE_COUNT = 12
SQUARES = 64


ALLOWED_ABLATIONS: tuple[str, ...] = SHARED_ABLATIONS + (
    "shuffle_squares",       # destroys per-square structure; primitive falsifier
    "permute_piece_types",   # destroys per-piece-type embedding alignment
    "zero_accumulator",      # forwards zero state — equivalent to dropping IDL entirely
)


def _sparse_accumulator(
    piece_planes: torch.Tensor, embedding: nn.Parameter
) -> torch.Tensor:
    """Sparse linear sum over occupied piece-squares.

    Args:
        piece_planes: ``(B, 12, 64)`` indicator tensor (0/1) for each
            (piece-type, plane-square) cell from ``simple_18``.
        embedding: ``(12, 64, d)`` parameter — one learned vector per
            (piece-type, plane-square) cell.

    Returns:
        ``(B, d)`` accumulator state.
    """
    # einsum semantics: S_b = sum_{t, s} planes[b, t, s] * embedding[t, s, :]
    return torch.einsum("bts,tsd->bd", piece_planes, embedding)


class IncrementalDeltaLinearHead(nn.Module):
    """p025 — Incremental Delta-Linear accumulator over the i193 trunk.

    The head learns a per-(piece-type, plane-square) embedding table of size
    ``(12, 64, d)`` and computes the sparse sum over occupied cells. The
    resulting accumulator state ``S`` is concatenated with the four standard
    trunk diagnostics (``gate``, ``gate_entropy``, ``mechanism_energy``,
    ``stream_disagreement``) and a small MLP produces a scalar delta logit.
    A sigmoid gate over the same fusion vector keeps the head from polluting
    the base logit on positions where IDL has no signal to add.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters (i193 baseline).
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # IDL accumulator hyper-parameters.
        accumulator_dim: int = 48,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -1.5,
        embedding_init_scale: float = 0.05,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "IncrementalDeltaLinearHead supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "IncrementalDeltaLinearHead requires the simple_18 board tensor"
            )
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        if int(accumulator_dim) < 1:
            raise ValueError("accumulator_dim must be >= 1")

        self.num_classes = 1
        self.ablation = str(ablation)
        self.accumulator_dim = int(accumulator_dim)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.trunk = build_trunk_from_kwargs(
            input_channels=int(input_channels),
            trunk_channels=int(trunk_channels),
            trunk_hidden_dim=int(trunk_hidden_dim),
            trunk_depth=int(trunk_depth),
            trunk_dropout=float(trunk_dropout),
            trunk_use_batchnorm=bool(trunk_use_batchnorm),
            trunk_gate_dim=trunk_gate_dim,
            trunk_ablation=str(trunk_ablation),
        )

        # Learned embedding table: one vector per (piece-type, plane-square).
        self.piece_square_embedding = nn.Parameter(
            torch.randn(PIECE_PLANE_COUNT, SQUARES, self.accumulator_dim)
            * float(embedding_init_scale)
        )
        self.accumulator_norm = nn.LayerNorm(self.accumulator_dim)

        # Fusion input: accumulator state + 4 trunk diagnostics + |S|.
        fusion_in = self.accumulator_dim + 4 + 1
        self._fusion_dim = fusion_in
        self.delta_mlp = small_mlp(
            fusion_in,
            int(head_hidden_dim),
            1,
            dropout=float(head_dropout),
        )
        self.gate_mlp = small_mlp(
            fusion_in,
            int(head_hidden_dim),
            1,
            dropout=float(head_dropout),
            final_bias_init=float(gate_init),
        )

    def _compute_accumulator(self, board: torch.Tensor) -> torch.Tensor:
        # Use only piece planes (channels 0..11) and flatten to (B, 12, 64).
        piece_planes = board[:, :PIECE_PLANE_COUNT].flatten(2).clamp(0.0, 1.0)
        if self.ablation == "shuffle_squares":
            # Destroy per-square structure by random column permutation.
            if piece_planes.shape[-1] > 1:
                perm = torch.randperm(piece_planes.shape[-1], device=piece_planes.device)
                piece_planes = piece_planes[..., perm]
        if self.ablation == "permute_piece_types":
            if piece_planes.shape[1] > 1:
                perm = torch.randperm(piece_planes.shape[1], device=piece_planes.device)
                piece_planes = piece_planes[:, perm]
        state = _sparse_accumulator(piece_planes, self.piece_square_embedding)
        if self.ablation == "zero_accumulator":
            state = torch.zeros_like(state)
        return state

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_output = self.trunk(board)
        base_logit = trunk_output["logits"]

        state = self._compute_accumulator(board)
        normalised = self.accumulator_norm(state)
        diagnostics = extract_trunk_diagnostics(trunk_output)
        norm_scalar = normalised.pow(2).mean(dim=-1, keepdim=True).clamp_min(1.0e-8).sqrt()
        fusion_in = torch.cat([normalised, diagnostics, norm_scalar], dim=1)

        delta_raw = self.delta_mlp(fusion_in).view(-1)
        gate_logit = self.gate_mlp(fusion_in).view(-1)
        gate = torch.sigmoid(gate_logit)

        logits, primitive_delta, effective_gate = fuse_with_base_logit(
            base_logit,
            gate,
            delta_raw,
            zero_delta=self.ablation in {"zero_delta", "trunk_only", "zero_accumulator"},
            force_gate_one=self.ablation == "disable_gate",
        )

        extra: dict[str, torch.Tensor] = {
            "idl_accumulator_norm": norm_scalar.view(-1),
            "idl_accumulator_state_l2": state.pow(2).mean(dim=-1).sqrt(),
            "idl_active_cells": (board[:, :PIECE_PLANE_COUNT].flatten(2) > 0.5)
            .float()
            .sum(dim=(1, 2)),
        }
        return standard_diagnostics_dict(
            trunk_output=trunk_output,
            logits=logits,
            base_logit=base_logit,
            primitive_delta=primitive_delta,
            delta_raw=delta_raw,
            gate=effective_gate,
            gate_logit=gate_logit,
            extra=extra,
        )


def build_incremental_delta_linear_head_from_config(
    config: dict[str, Any],
) -> IncrementalDeltaLinearHead:
    cfg = dict(config)
    return IncrementalDeltaLinearHead(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        accumulator_dim=int(cfg.get("accumulator_dim", 48)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -1.5)),
        embedding_init_scale=float(cfg.get("embedding_init_scale", 0.05)),
        ablation=str(cfg.get("ablation", "none")),
    )
