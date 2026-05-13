"""Promotion-Aware Head (i246) — PFCT primitive integrated with the i193 trunk.

The Promotion-Fanout Counterfactual Tensor (PFCT) primitive enumerates the
four chess-legal promotion-piece substitutions {Q, R, B, N} for every own
near-promotion pawn (rank 7 for white-to-move, rank 2 for black-to-move),
runs the substituted boards back through the shared i193 trunk to obtain a
``(|P_near|, 4, d)`` fanout tensor of joint pool features, and pools each
pawn's fanout with a cross-attention head conditioned on the baseline trunk
feature and per-pawn square / piece-type embeddings. A learned gate (forced
to zero whenever the position has no near-promotion pawn) injects the
per-pawn delta into the i193 base logit:

    final_logit = base_logit + gate * sum_p pawn_delta(p)

CRTK metadata, source labels, verification flags, and engine scores are
**not** consulted. The substitutions are computed analytically on the
``simple_18`` piece planes (which already encode placement, side-to-move,
castling rights, and en-passant) using chess-rule-derived index logic; no
``python-chess`` call is required inside the forward pass.

Spec sources:

- ``ideas/research/primitives/claude_03_promotion_fanout_counterfactual.md``
- ``ideas/research/primitives/prototypes/pfct_prototype.py``
- ``ideas/research/primitives/PRIMITIVE_TRAINING_TODO.md``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


# simple_18 plane layout (board_features.PIECE_PLANES):
#   0: white pawn  1: white knight  2: white bishop  3: white rook
#   4: white queen 5: white king
#   6: black pawn  7: black knight  8: black bishop  9: black rook
#   10: black queen 11: black king
#   12: white_to_move
PIECE_PLANE_COUNT = 12
WHITE_PAWN_PLANE = 0
BLACK_PAWN_PLANE = 6
STM_CHANNEL = 12
SQUARES = 64

# Promotion enumeration: order matches the PFCT prototype
# (Q -> R -> B -> N) so prototype tests and visualisations line up.
PROMOTION_TYPE_NAMES: tuple[str, ...] = ("Q", "R", "B", "N")
PROMOTION_TYPE_COUNT = len(PROMOTION_TYPE_NAMES)

# Plane offsets for the promoted piece in {Q, R, B, N} order:
#   simple_18 piece order within a colour is (P=0, N=1, B=2, R=3, Q=4, K=5)
#   So Q=4, R=3, B=2, N=1 for white. Add 6 for black.
PROMOTION_PLANE_OFFSETS_WHITE: tuple[int, ...] = (4, 3, 2, 1)
PROMOTION_PLANE_OFFSETS_BLACK: tuple[int, ...] = (10, 9, 8, 7)

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "copy_baseline_fanout",     # primary A1 falsifier: 4 copies of baseline feature
    "uniform_attention",        # disable learned attention (fixed 1/4 weighting)
    "zero_delta",               # primitive_delta == 0 everywhere
    "force_open_gate",          # bypass gate (gate = 1 if pawns present)
    "trunk_only",               # equivalent to disabling the primitive entirely
)


def _build_promote_plane_lookup() -> torch.Tensor:
    """(2, 4) long tensor mapping (own_color, promotion_type) -> simple_18 plane.

    ``own_color = 0`` is white-to-move, ``own_color = 1`` is black-to-move.
    """
    return torch.tensor(
        [
            list(PROMOTION_PLANE_OFFSETS_WHITE),
            list(PROMOTION_PLANE_OFFSETS_BLACK),
        ],
        dtype=torch.long,
    )


@dataclass(frozen=True)
class NearPromotionSlots:
    """Compact representation of up to ``K`` near-promotion pawn slots per sample."""

    source_square: torch.Tensor   # (B, K) long, simple_18 plane square index
    promote_square: torch.Tensor  # (B, K) long, simple_18 plane square index
    own_pawn_plane: torch.Tensor  # (B, K) long, 0 (white) or 6 (black)
    own_color: torch.Tensor       # (B, K) long, 0 (white) or 1 (black)
    valid: torch.Tensor           # (B, K) bool, True for real pawn slots


def _plane_square_for_simple18(rank_idx: torch.Tensor, file_idx: torch.Tensor) -> torch.Tensor:
    """Compute the flat (rank * 8 + file) square index used by ``board.flatten(2)``.

    simple_18 stores plane[rank=0] at the top of the board (rank 8 for white,
    rank 1 for black is plane row 7). The "square" returned here is the row-major
    index into the (8, 8) plane, not the chess square number — that is the
    convention used by ``board.flatten(2)`` throughout the trunk.
    """
    return rank_idx * 8 + file_idx


def find_near_promotion_slots(board: torch.Tensor, max_pawns: int) -> NearPromotionSlots:
    """Identify up to ``max_pawns`` own near-promotion pawn slots per sample.

    For white-to-move, scans the row of plane 0 at rank-index 1 (rank 7);
    for black-to-move, scans the row of plane 6 at rank-index 6 (rank 2).
    Slots are filled in file order (a-file first). When there are fewer
    near-promotion pawns than ``max_pawns``, the remaining slots are marked
    invalid and point at a dummy square that will be ignored downstream.

    Args:
        board: ``(B, 18, 8, 8)`` simple_18 board tensor.
        max_pawns: maximum number of slots per sample (a fixed K for batching).

    Returns:
        ``NearPromotionSlots`` with shape ``(B, max_pawns)`` everywhere.
    """
    if max_pawns < 1:
        raise ValueError("max_pawns must be >= 1")
    batch = board.shape[0]
    device = board.device

    stm = board[:, STM_CHANNEL].mean(dim=(1, 2)).clamp(0.0, 1.0)  # (B,)
    is_white = stm > 0.5  # (B,) bool

    # Per-file presence indicators (1.0 if an own pawn is near promotion).
    white_near = (board[:, WHITE_PAWN_PLANE, 1, :] > 0.5).to(board.dtype)  # (B, 8)
    black_near = (board[:, BLACK_PAWN_PLANE, 6, :] > 0.5).to(board.dtype)  # (B, 8)
    own_near = torch.where(is_white.unsqueeze(-1), white_near, black_near)  # (B, 8)

    # Tie-break by file index (a-file first). Score keeps presence as the dominant
    # term while breaking ties in favour of low-file pawns.
    file_index = torch.arange(8, device=device, dtype=board.dtype)
    presence_score = own_near * (8.0 - file_index).unsqueeze(0)  # (B, 8)
    k = min(int(max_pawns), 8)
    top_scores, top_files = presence_score.topk(k, dim=1)
    valid = top_scores > 0.5  # (B, K) bool

    if k < int(max_pawns):
        pad_files = top_files.new_zeros(batch, int(max_pawns) - k)
        pad_valid = valid.new_zeros(batch, int(max_pawns) - k)
        top_files = torch.cat([top_files, pad_files], dim=1)
        valid = torch.cat([valid, pad_valid], dim=1)

    own_color = torch.where(
        is_white.unsqueeze(-1),
        torch.zeros_like(top_files),
        torch.ones_like(top_files),
    )  # (B, K) 0 white, 1 black

    # plane row indices for source (white rank 7 -> row 1, black rank 2 -> row 6)
    src_rank_idx = torch.where(
        own_color == 0,
        torch.ones_like(top_files),
        torch.full_like(top_files, 6),
    )
    promote_rank_idx = torch.where(
        own_color == 0,
        torch.zeros_like(top_files),
        torch.full_like(top_files, 7),
    )
    source_square = _plane_square_for_simple18(src_rank_idx, top_files)
    promote_square = _plane_square_for_simple18(promote_rank_idx, top_files)
    own_pawn_plane = torch.where(
        own_color == 0,
        torch.full_like(top_files, WHITE_PAWN_PLANE),
        torch.full_like(top_files, BLACK_PAWN_PLANE),
    )

    return NearPromotionSlots(
        source_square=source_square.long(),
        promote_square=promote_square.long(),
        own_pawn_plane=own_pawn_plane.long(),
        own_color=own_color.long(),
        valid=valid,
    )


def build_promotion_counterfactuals(
    board: torch.Tensor,
    slots: NearPromotionSlots,
    promote_plane_lookup: torch.Tensor,
) -> torch.Tensor:
    """Build all ``(K, 4)`` promotion-substituted boards per sample.

    For each valid slot, the source pawn is removed and a promoted piece is
    placed on the corresponding promotion square. Any existing piece on the
    promotion square (e.g. an enemy piece for a capture-promotion) is cleared
    first so the result is a legal one-hot piece encoding. Invalid slots
    pass the original board through unchanged; downstream code masks their
    contributions via ``slots.valid``.

    Returns:
        Tensor of shape ``(B, K, 4, 18, 8, 8)``.
    """
    board = board.contiguous()
    batch, channels, height, width = board.shape
    if channels != PIECE_PLANE_COUNT + 6 or (height, width) != (8, 8):
        raise ValueError(f"Expected simple_18 board (B, 18, 8, 8), got {tuple(board.shape)}")
    K = slots.source_square.shape[1]
    device = board.device
    dtype = board.dtype

    x_flat = board.flatten(2)  # (B, 18, 64)

    # Plane / square one-hots used to compose the additive/removal masks.
    src_plane_oh = F.one_hot(slots.own_pawn_plane, num_classes=channels).to(dtype=dtype)
    # (B, K, 18)
    src_sq_oh = F.one_hot(slots.source_square.clamp(0, SQUARES - 1), num_classes=SQUARES).to(dtype=dtype)
    # (B, K, 64)
    promote_sq_oh = F.one_hot(slots.promote_square.clamp(0, SQUARES - 1), num_classes=SQUARES).to(dtype=dtype)
    # (B, K, 64)
    valid_f = slots.valid.to(dtype=dtype)  # (B, K)

    # Promotion-plane lookup for each (sample, slot, type).
    # promote_plane_lookup: (2, 4) -> own_color, type -> plane offset
    promote_planes_per_slot = promote_plane_lookup.to(device=device)[slots.own_color]
    # (B, K, 4) where each entry is the plane index for that promotion type.
    promote_plane_oh = F.one_hot(promote_planes_per_slot, num_classes=channels).to(dtype=dtype)
    # (B, K, 4, 18)

    # --- Masks ---
    # remove_source_mask: clear the pawn from its source plane / square.
    remove_source_mask = src_plane_oh.unsqueeze(-1) * src_sq_oh.unsqueeze(-2)
    remove_source_mask = remove_source_mask * valid_f.unsqueeze(-1).unsqueeze(-1)
    # (B, K, 18, 64)
    remove_source_mask = remove_source_mask.unsqueeze(2)  # (B, K, 1, 18, 64) broadcasts over 4 types

    # remove_target_mask: clear any piece plane (0..11) at the promotion square.
    piece_plane_indicator = board.new_zeros(channels)
    piece_plane_indicator[:PIECE_PLANE_COUNT] = 1.0
    # (B, K, 18, 64) — same for all 4 promotion types.
    remove_target_mask = (
        piece_plane_indicator.view(1, 1, channels, 1)
        * promote_sq_oh.unsqueeze(-2)
        * valid_f.unsqueeze(-1).unsqueeze(-1)
    )
    remove_target_mask = remove_target_mask.unsqueeze(2)

    # Removal is the union of both masks (1 where any cell should be zeroed).
    removal_mask = torch.maximum(remove_source_mask, remove_target_mask)
    # (B, K, 1, 18, 64)

    # add_promotion_mask: place the promoted piece on the promotion square,
    # per (sample, slot, type).
    add_mask = (
        promote_plane_oh.unsqueeze(-1)            # (B, K, 4, 18, 1)
        * promote_sq_oh.unsqueeze(2).unsqueeze(-2)  # (B, K, 1, 1, 64)
        * valid_f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1, 1)
    )
    # (B, K, 4, 18, 64)

    # Expand the base board across slot/type axes and apply masks. The order is:
    #   1. remove source pawn and any current piece on promote square
    #   2. add the promoted piece on promote square (always 0/1)
    x_expand = x_flat.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 18, 64)
    cf_flat = x_expand * (1.0 - removal_mask) + add_mask
    # Clamp away from any tiny FP error to keep the encoding boolean-clean.
    cf_flat = cf_flat.clamp(0.0, 1.0)
    cf = cf_flat.view(batch, K, PROMOTION_TYPE_COUNT, channels, height, width)
    return cf


def _trunk_joint_features(
    trunk: ExchangeThenKingDualStreamNetwork,
    board: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate i193's joint feature path without re-running the final logit.

    Returns ``(joint, ex_pool, kg_pool)``. The trunk's own forward is still
    called separately for the baseline pass — this helper exists so the same
    pool/joint feature can be extracted for counterfactual boards without
    paying for the trunk's head MLPs four times per pawn.
    """
    feats = trunk.feature_builder(board)
    if trunk.ablation == "shared_stream_only":
        ex_input = board
        kg_input = board
    else:
        ex_input = torch.cat([board, feats.exchange], dim=1)
        kg_input = torch.cat([board, feats.king], dim=1)
    _, ex_pool = trunk.exchange_encoder(ex_input)
    if trunk.ablation == "shared_stream_only":
        kg_pool = ex_pool
    else:
        _, kg_pool = trunk.king_encoder(kg_input)
    joint = torch.cat([ex_pool, kg_pool, feats.summary], dim=1)
    return joint, ex_pool, kg_pool


class PromotionAwareHead(nn.Module):
    """i246 — Promotion-Aware Head over the i193 dual-stream trunk.

    Forward pass:

    1. Run the i193 trunk once on the input board to obtain the base logit
       and a joint pool feature ``f_base`` (i193 pool concat + summary).
    2. Pick the top-K own near-promotion pawn squares per sample.
    3. Build the ``(B, K, 4, 18, 8, 8)`` counterfactual grid and pass it
       through the shared trunk encoder to obtain the fanout features
       ``F(p)`` of shape ``(B, K, 4, d)``.
    4. For each pawn, compute a per-piece-type cross-attention pool
       (queries from baseline + pawn square embedding, keys/values from
       fanout + promotion-type embedding) to obtain ``pawn_feature``.
    5. Project each pawn's pooled feature to a scalar delta, mask invalid
       slots, sum over pawns to get ``primitive_delta``, and gate the
       result with ``sigmoid(MLP_gate(f_base)) * has_promotion_pawn``.
    6. ``final_logit = base_logit + gated_delta``.

    Ablations are exposed for the standard falsifier checks documented in
    ``ideas/research/primitives/claude_03_promotion_fanout_counterfactual.md``.
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
        # PFCT head hyper-parameters.
        max_promotion_pawns: int = 4,
        pawn_embed_dim: int = 32,
        promotion_embed_dim: int = 16,
        attn_dim: int = 64,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "PromotionAwareHead supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != 18:
            raise ValueError(
                "PromotionAwareHead requires the simple_18 board tensor"
            )
        if int(max_promotion_pawns) < 1 or int(max_promotion_pawns) > 8:
            raise ValueError("max_promotion_pawns must be between 1 and 8")
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )
        self.num_classes = 1
        self.ablation = str(ablation)
        self.max_promotion_pawns = int(max_promotion_pawns)
        self.spec = BoardTensorSpec(input_channels=int(input_channels))

        self.trunk = ExchangeThenKingDualStreamNetwork(
            input_channels=int(input_channels),
            num_classes=1,
            channels=int(trunk_channels),
            hidden_dim=int(trunk_hidden_dim),
            depth=int(trunk_depth),
            dropout=float(trunk_dropout),
            use_batchnorm=bool(trunk_use_batchnorm),
            gate_dim=trunk_gate_dim,
            ablation=str(trunk_ablation),
        )

        self.feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        self.pawn_embed_dim = int(pawn_embed_dim)
        self.promotion_embed_dim = int(promotion_embed_dim)
        self.attn_dim = int(attn_dim)

        # Per-square pawn embedding: indexed by simple_18 plane square (0..63).
        # Only 16 squares are ever active (8 white rank-7, 8 black rank-2), but
        # we keep a full 64-row table for the simplest indexing.
        self.pawn_square_embed = nn.Embedding(SQUARES, self.pawn_embed_dim)
        # Per-promotion-type embedding (Q, R, B, N).
        self.promotion_type_embed = nn.Embedding(PROMOTION_TYPE_COUNT, self.promotion_embed_dim)

        head_dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        # Query MLP: baseline feature concatenated with pawn-square embedding.
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim + self.pawn_embed_dim),
            nn.Linear(self.feature_dim + self.pawn_embed_dim, self.attn_dim),
            nn.GELU(),
        )
        # Key MLP: counterfactual joint feature concatenated with promotion-type embedding.
        self.key_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim + self.promotion_embed_dim),
            nn.Linear(self.feature_dim + self.promotion_embed_dim, self.attn_dim),
            nn.GELU(),
        )
        # Value projection: counterfactual feature down to head_hidden_dim.
        self.value_proj = nn.Linear(self.feature_dim, int(head_hidden_dim))
        # Per-pawn delta head.
        self.delta_head = nn.Sequential(
            nn.LayerNorm(int(head_hidden_dim)),
            nn.Linear(int(head_hidden_dim), max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            head_dropout_module,
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        # Gate MLP: feeds on the baseline feature only so the trunk decides when
        # promotion fanout matters.
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        # Initialise the gate near-closed so the primitive starts as a no-op.
        with torch.no_grad():
            final_layer = self.gate_head[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(float(gate_init))

        self.register_buffer(
            "promote_plane_lookup",
            _build_promote_plane_lookup(),
            persistent=False,
        )

    def _maybe_replace_fanout_with_baseline(
        self,
        cf_features: torch.Tensor,
        base_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the A1 ablation: replace counterfactual fanout with copies of baseline.

        Used to verify that the four-fold piece-type substitution actually
        adds signal beyond what the baseline trunk already encodes.
        """
        if self.ablation != "copy_baseline_fanout":
            return cf_features
        b, k, t, _ = cf_features.shape
        return base_feature.unsqueeze(1).unsqueeze(2).expand(b, k, t, self.feature_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]
        device = board.device
        dtype = board.dtype
        K = self.max_promotion_pawns

        # --- 1. baseline trunk forward (with all i193 diagnostics).
        base_out = self.trunk(board)
        base_logit = base_out["logits"].view(-1)
        base_joint, _, _ = _trunk_joint_features(self.trunk, board)

        # --- 2. identify near-promotion pawn slots per sample.
        slots = find_near_promotion_slots(board, max_pawns=K)
        valid = slots.valid  # (B, K) bool
        valid_f = valid.to(dtype=dtype)
        any_pawn = valid.any(dim=1).to(dtype=dtype)  # (B,)

        # --- 3. build counterfactual boards and run shared trunk encoder.
        cf_grid = build_promotion_counterfactuals(board, slots, self.promote_plane_lookup)
        # (B, K, 4, 18, 8, 8)
        cf_flat = cf_grid.view(batch * K * PROMOTION_TYPE_COUNT, *cf_grid.shape[3:])
        cf_joint, _, _ = _trunk_joint_features(self.trunk, cf_flat)
        cf_features = cf_joint.view(batch, K, PROMOTION_TYPE_COUNT, self.feature_dim)

        # Apply A1 ablation if requested. After this point ``cf_features``
        # is the per-(pawn, type) "phi" feature used by attention.
        cf_features = self._maybe_replace_fanout_with_baseline(cf_features, base_joint)

        # --- 4. cross-attention per pawn, over promotion piece types.
        pawn_emb = self.pawn_square_embed(slots.source_square)  # (B, K, pawn_embed_dim)
        type_idx = torch.arange(PROMOTION_TYPE_COUNT, device=device, dtype=torch.long)
        type_emb_table = self.promotion_type_embed(type_idx)  # (4, promotion_embed_dim)
        type_emb = type_emb_table.view(1, 1, PROMOTION_TYPE_COUNT, self.promotion_embed_dim).expand(
            batch, K, PROMOTION_TYPE_COUNT, self.promotion_embed_dim
        )

        base_for_query = base_joint.unsqueeze(1).expand(batch, K, self.feature_dim)
        query_input = torch.cat([base_for_query, pawn_emb], dim=-1)
        query = self.query_proj(query_input)  # (B, K, attn_dim)
        key_input = torch.cat([cf_features, type_emb], dim=-1)
        key = self.key_proj(key_input)  # (B, K, 4, attn_dim)

        attn_scale = math.sqrt(self.attn_dim)
        attn_logits = torch.einsum("bki,bkti->bkt", query, key) / attn_scale
        # (B, K, 4)
        if self.ablation == "uniform_attention":
            attn_weights = attn_logits.new_full(attn_logits.shape, 1.0 / PROMOTION_TYPE_COUNT)
        else:
            attn_weights = torch.softmax(attn_logits, dim=-1)

        values = self.value_proj(cf_features)  # (B, K, 4, head_hidden_dim)
        pawn_feature = torch.einsum("bkt,bkth->bkh", attn_weights, values)
        # (B, K, head_hidden_dim)

        # Mask out invalid slots so they cannot contribute to the delta.
        pawn_feature = pawn_feature * valid_f.unsqueeze(-1)
        pawn_delta = self.delta_head(pawn_feature).view(batch, K)
        pawn_delta = pawn_delta * valid_f

        primitive_delta = pawn_delta.sum(dim=1)
        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(primitive_delta)

        # --- 5. gate and final logit.
        gate_logit = self.gate_head(base_joint).view(-1)
        gate = torch.sigmoid(gate_logit)
        # The gate is structurally forced to zero on positions with no
        # near-promotion pawn, matching the spec's "zero overhead" guarantee.
        gate = gate * any_pawn
        if self.ablation == "force_open_gate":
            gate = torch.where(any_pawn > 0, torch.ones_like(gate), torch.zeros_like(gate))
        gated_delta = gate * primitive_delta
        logits = base_logit + gated_delta

        # --- 6. diagnostics.
        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        attn_entropy = -(
            attn_weights.clamp_min(1.0e-8).log() * attn_weights
        ).sum(dim=-1)  # (B, K)
        attn_entropy = (attn_entropy * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        attn_entropy = attn_entropy / math.log(PROMOTION_TYPE_COUNT)

        argmax_type = attn_weights.argmax(dim=-1)  # (B, K) long, 0..3
        # Per-sample dominant promotion type for the first valid pawn — a useful
        # diagnostic for slice reports. -1 if no near-promotion pawn.
        first_valid = valid.float() * (1.0 - torch.arange(K, device=device, dtype=dtype) * 1.0e-3)
        first_valid_idx = first_valid.argmax(dim=1)
        # Gather using first_valid_idx but fall back to -1 if no pawn at all.
        dominant_type = argmax_type.gather(1, first_valid_idx.unsqueeze(-1)).squeeze(-1)
        dominant_type = torch.where(
            any_pawn > 0,
            dominant_type,
            torch.full_like(dominant_type, -1),
        )

        # Fanout magnitude: ||F(p, T) - mean_T F(p, T)|| summed per pawn,
        # then mean-pooled over valid pawns.
        fanout_mean = cf_features.mean(dim=2, keepdim=True)  # (B, K, 1, F)
        fanout_residual = cf_features - fanout_mean  # (B, K, 4, F)
        fanout_norm = fanout_residual.pow(2).sum(dim=-1).sqrt()  # (B, K, 4)
        fanout_norm = fanout_norm.mean(dim=-1)  # (B, K)
        fanout_norm = (fanout_norm * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": logits,
            "base_logit": base_logit,
            "primitive_delta": primitive_delta,
            "primitive_delta_raw": pawn_delta.sum(dim=1),
            "primitive_gate": gate,
            "primitive_gate_logit": gate_logit,
            "primitive_gate_entropy": gate_entropy,
            "primitive_logit_contribution": gated_delta,
            "promotion_pawn_count": valid_f.sum(dim=1),
            "promotion_has_pawn": any_pawn,
            "promotion_attention_entropy": attn_entropy,
            "promotion_dominant_type": dominant_type.to(dtype=dtype),
            "promotion_fanout_dispersion": fanout_norm,
            "promotion_pawn_delta_max": pawn_delta.abs().amax(dim=1)
            if K > 0
            else pawn_delta.new_zeros(batch),
            "mechanism_energy": base_out["mechanism_energy"]
            + fanout_norm.detach(),
            "proposal_profile_strength": (gated_delta.abs() * gate_entropy).clamp(0.0, 20.0),
            "proposal_keyword_count": logits.new_full(
                (batch,), float(self.feature_dim + self.attn_dim)
            ),
        }
        for key, value in base_out.items():
            if key == "logits":
                continue
            diag_key = key if key.startswith("base_") or key in {"gate", "gate_logit"} else f"trunk_{key}"
            diagnostics.setdefault(diag_key, value)
        return diagnostics


def build_promotion_aware_head_from_config(config: dict[str, Any]) -> PromotionAwareHead:
    cfg = dict(config)
    return PromotionAwareHead(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim", None)),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        max_promotion_pawns=int(cfg.get("max_promotion_pawns", 4)),
        pawn_embed_dim=int(cfg.get("pawn_embed_dim", 32)),
        promotion_embed_dim=int(cfg.get("promotion_embed_dim", 16)),
        attn_dim=int(cfg.get("attn_dim", 64)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )
