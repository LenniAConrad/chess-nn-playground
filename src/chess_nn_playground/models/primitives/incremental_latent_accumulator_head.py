"""Incremental Latent Accumulator Head (p028).

Promotes the **Incremental Latent Accumulator (ILA)** primitive (first-ranked
proposal of
``ideas/research/primitives/external_24_incremental_latent_accumulator_directional_scan.md``).
ILA generalises the NNUE HalfKA accumulator into a deep operator: a global
latent vector ``h`` is composed as the sum of per-feature embeddings, then a
non-linear projection ``phi`` produces a normalised representation.

    h = sum_{(t, s) : x_{t, s} == 1} E_{t, s}    (sparse linear sum)
    z = LayerNorm(phi(h))                        (non-linear lift)

The math is identical to the IDL primitive at the linear stage, but ILA adds
the ``phi`` non-linearity and an explicit LayerNorm so the head is *not* a
pure linear accumulator. This separates the "linear delta" claim (covered by
``p025``) from the "non-linear latent delta propagates through" claim (the
subject of ILA).

To honour the king-anchored HalfKA refinement called out by NNUE we use a
**two-stream** accumulator: a global piece-square embedding for the own
side plus a separate king-anchored embedding indexed by the
``(own_king_square, piece_type, square)`` triple (rule-derived from
``simple_18``). The own-king and enemy-king squares are read directly from
the simple_18 piece planes — never from any metadata.

Deferred external_24 proposals (research-only): ``LMTG`` legal-move topology
gate (covered by ``p027``), ``BPDO`` bit-population differentiable operator
(a counting layer with a different gradient profile — outside the scope of
this batch), ``SEI`` symmetry-equivariant involution (a weight-tying
primitive on the trunk, not a head), ``DSS`` directional stopping scan
(covered by ``p029`` / ``p030``).
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
STM_CHANNEL = 12
WHITE_KING_PLANE = 5
BLACK_KING_PLANE = 11
KING_ANCHOR_DUMMY = SQUARES  # dummy index used when the king is missing
KING_ANCHOR_TABLE_SIZE = SQUARES + 1


ALLOWED_ABLATIONS: tuple[str, ...] = SHARED_ABLATIONS + (
    "zero_global_accumulator",
    "zero_king_accumulator",
    "linear_only",            # drop the phi non-linearity (math should collapse to IDL)
    "shuffle_square_order",   # permute square indices for falsification
)


def _own_king_square(board: torch.Tensor) -> torch.Tensor:
    """Return the plane-square index of the own king per sample.

    Falls back to ``SQUARES`` (the dummy anchor row) on positions where the
    own king is absent, which should not happen on legal data but keeps the
    indexing well-defined under shuffled / corrupted batches.
    """
    stm = board[:, STM_CHANNEL].mean(dim=(1, 2)).clamp(0.0, 1.0)  # (B,)
    is_white = stm > 0.5
    white_king = board[:, WHITE_KING_PLANE].flatten(1).clamp(0.0, 1.0)  # (B, 64)
    black_king = board[:, BLACK_KING_PLANE].flatten(1).clamp(0.0, 1.0)
    own_king = torch.where(is_white.unsqueeze(-1), white_king, black_king)
    has_king = own_king.sum(dim=-1) > 0.5
    king_index = own_king.argmax(dim=-1)  # (B,) long
    dummy = torch.full_like(king_index, KING_ANCHOR_DUMMY)
    return torch.where(has_king, king_index, dummy).long()


class IncrementalLatentAccumulatorHead(nn.Module):
    """p028 — Incremental Latent Accumulator on the i193 dual-stream trunk.

    Forward pass:

    1. Read the simple_18 piece planes and the own-king square (rule-exact).
    2. Sparse-sum a learned ``(12, 64, d)`` global embedding to get
       ``h_global``.
    3. Sparse-sum a learned ``(65, 12, 64, d')`` king-anchored embedding to
       get ``h_king``, where the first axis is the own-king square (with the
       last row reserved for "no own king").
    4. Concatenate ``h_global`` and ``h_king``, apply the ``phi`` MLP and a
       LayerNorm to obtain the non-linear latent.
    5. Fuse with the trunk diagnostics and produce the additive-gated delta.
    """

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # ILA hyper-parameters.
        global_dim: int = 48,
        king_dim: int = 16,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        embedding_init_scale: float = 0.05,
        gate_init: float = -1.5,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "IncrementalLatentAccumulatorHead supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "IncrementalLatentAccumulatorHead requires the simple_18 board tensor"
            )
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        if int(global_dim) < 1 or int(king_dim) < 1:
            raise ValueError("global_dim and king_dim must both be >= 1")

        self.num_classes = 1
        self.ablation = str(ablation)
        self.global_dim = int(global_dim)
        self.king_dim = int(king_dim)
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

        # Global per-(piece-type, square) embedding (size (12, 64, d)).
        self.global_embedding = nn.Parameter(
            torch.randn(PIECE_PLANE_COUNT, SQUARES, self.global_dim)
            * float(embedding_init_scale)
        )
        # King-anchored per-(king-square, piece-type, square) embedding
        # (size (65, 12, 64, d')) — the +1 is the "no own king" dummy row.
        self.king_embedding = nn.Parameter(
            torch.randn(KING_ANCHOR_TABLE_SIZE, PIECE_PLANE_COUNT, SQUARES, self.king_dim)
            * float(embedding_init_scale)
        )

        latent_in = self.global_dim + self.king_dim
        self.phi_mlp = nn.Sequential(
            nn.LayerNorm(latent_in),
            nn.Linear(latent_in, max(latent_in, int(head_hidden_dim))),
            nn.GELU(),
            nn.Linear(max(latent_in, int(head_hidden_dim)), latent_in),
        )
        self.latent_norm = nn.LayerNorm(latent_in)

        fusion_in = latent_in + 4
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

    def _compute_accumulators(
        self, board: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        piece_planes = board[:, :PIECE_PLANE_COUNT].flatten(2).clamp(0.0, 1.0)  # (B, 12, 64)
        if self.ablation == "shuffle_square_order" and piece_planes.shape[-1] > 1:
            perm = torch.randperm(piece_planes.shape[-1], device=piece_planes.device)
            piece_planes = piece_planes[..., perm]
        # Global sparse sum.
        h_global = torch.einsum("bts,tsd->bd", piece_planes, self.global_embedding)
        if self.ablation == "zero_global_accumulator":
            h_global = torch.zeros_like(h_global)

        king_idx = _own_king_square(board)  # (B,) long in [0, 65)
        king_subset = self.king_embedding[king_idx]  # (B, 12, 64, king_dim)
        h_king = torch.einsum("bts,btsd->bd", piece_planes, king_subset)
        if self.ablation == "zero_king_accumulator":
            h_king = torch.zeros_like(h_king)
        return h_global, h_king, king_idx

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        trunk_output = self.trunk(board)
        base_logit = trunk_output["logits"]

        h_global, h_king, king_idx = self._compute_accumulators(board)
        h_concat = torch.cat([h_global, h_king], dim=-1)
        if self.ablation == "linear_only":
            latent = self.latent_norm(h_concat)
        else:
            latent = self.latent_norm(self.phi_mlp(h_concat))

        diagnostics = extract_trunk_diagnostics(trunk_output)
        fusion_in = torch.cat([latent, diagnostics], dim=1)

        delta_raw = self.delta_mlp(fusion_in).view(-1)
        gate_logit = self.gate_mlp(fusion_in).view(-1)
        gate = torch.sigmoid(gate_logit)
        logits, primitive_delta, effective_gate = fuse_with_base_logit(
            base_logit,
            gate,
            delta_raw,
            zero_delta=self.ablation in {"zero_delta", "trunk_only"},
            force_gate_one=self.ablation == "disable_gate",
        )

        active_cells = (board[:, :PIECE_PLANE_COUNT].flatten(2) > 0.5).float().sum(dim=(1, 2))
        extra: dict[str, torch.Tensor] = {
            "ila_global_norm": h_global.pow(2).mean(dim=-1).sqrt(),
            "ila_king_norm": h_king.pow(2).mean(dim=-1).sqrt(),
            "ila_latent_norm": latent.pow(2).mean(dim=-1).sqrt(),
            "ila_active_cells": active_cells,
            "ila_king_index": king_idx.to(dtype=board.dtype),
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


def build_incremental_latent_accumulator_head_from_config(
    config: dict[str, Any],
) -> IncrementalLatentAccumulatorHead:
    cfg = dict(config)
    return IncrementalLatentAccumulatorHead(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        global_dim=int(cfg.get("global_dim", 48)),
        king_dim=int(cfg.get("king_dim", 16)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        embedding_init_scale=float(cfg.get("embedding_init_scale", 0.05)),
        gate_init=float(cfg.get("gate_init", -1.5)),
        ablation=str(cfg.get("ablation", "none")),
    )
