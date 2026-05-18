"""Promotion and Underpromotion Geometry primitive (p052, PUGP).

Source: ``ideas/research/primitives/external_47_promotion_underpromotion_primitive.md``.

The primitive is an additive, gated logit-delta head on top of the i193
``ExchangeThenKingDualStreamNetwork`` trunk. It encodes
side-to-move-canonicalised promotion geometry as a fixed-shape feature
vector and projects that vector through a small MLP to a scalar
``primitive_delta``. The gate is a sigmoid MLP over the trunk joint pool
plus a small set of candidate-count summaries; its initial bias is
strongly negative so the primitive starts as a no-op and only opens when
the trunk feature plus the candidate set indicate a promotion-relevant
position.

Forward path

    1. ``simple_18`` board ``x`` is canonicalised to ``B^c = C(x)`` where
       ``C`` is the identity for white-to-move and a vertical mirror with
       white/black piece-plane swap (plus castling-plane swap and the
       en-passant plane flipped) for black-to-move. After ``C`` the own
       side always moves toward canonical row 0; near-promotion own
       pawns live at canonical row 1; the promotion rank is canonical
       row 0. The STM plane is fixed at 1.0 in canonical space.
    2. The canonical board feeds three deterministic geometry blocks:
        - **Global pawn-distance summary**: counts of own / opponent
          pawns at canonical rows 1, 2, 3.
        - **Per-file candidate masks**: ``push``, ``capture-left``,
          ``capture-right`` indicators for own near-promotion pawns,
          obtained from per-file occupancy of canonical row 0.
        - **Per-arrival-square attack / defence features**: for each
          of the 8 arrival squares on canonical row 0 and each
          promotion piece type ``t in {Q, R, B, N}`` the head computes
          the promoted piece's attack mask (via the shared
          ``ray_geometry`` lookup tables for Q/R/B and a precomputed
          knight template), the gives-check indicator
          ``c_t``, the king-zone overlap ``z_t``, attackers /
          defenders of ``u`` and the safety score ``s_t = clip(d - a,
          -4, 4) / 4``, plus the knight-fork hint
          ``kappa_N`` over enemy high-value targets.
    3. The per-arrival-square per-type features are aggregated with
       **per-file candidate masking** (so empty files contribute zero)
       and then **sum + max pooled** across files to a fixed-shape
       feature vector. Underpromotion is encoded as the explicit
       ``delta-to-queen`` differences ``Delta_R``, ``Delta_B``,
       ``Delta_N`` plus the dedicated ``kappa_N`` knight-fork hint.
    4. The pooled PUGP vector is concatenated with the i193 trunk
       joint pool feature (via :func:`trunk_joint_features`) and fed
       through a two-layer MLP to produce ``primitive_delta_raw``.
       The gate MLP consumes the joint feature plus the
       ``has_candidate`` indicator and the candidate-count summary;
       ``primitive_gate = sigmoid(gate_logit)``. Output:

           final_logit = base_logit + primitive_gate * primitive_delta_raw.

CRTK metadata, source labels, verification flags, engine evaluations
and principal variations are *not* consulted.

The primitive is a strict no-op when no candidate exists: the
candidate mask is zero, the pooled features are zero, and the trained
gate's bias keeps the gate near zero on those samples.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from chess_nn_playground.models.primitives.ray_geometry import (
    DIRECTIONS,
    NUM_DIRECTIONS,
    RAY_MAX_LEN,
    SQUARES,
    build_ray_step_index,
)
from chess_nn_playground.models.primitives.trunk_features import trunk_joint_features
from chess_nn_playground.models.trunk.exchange_then_king_dual_stream import (
    DualStreamFeatureBuilder,
    ExchangeThenKingDualStreamNetwork,
)
from chess_nn_playground.models.trunk.idea_blocks import BoardTensorSpec, require_board_tensor


NUM_PIECE_CHANNELS = 12
SIMPLE_18_PLANES = 18

# Canonical own piece plane order (post canonicalisation): P, N, B, R, Q, K.
OWN_PAWN = 0
OWN_KNIGHT = 1
OWN_BISHOP = 2
OWN_ROOK = 3
OWN_QUEEN = 4
OWN_KING = 5
OPP_PAWN = 6
OPP_KNIGHT = 7
OPP_BISHOP = 8
OPP_ROOK = 9
OPP_QUEEN = 10
OPP_KING = 11

# Promotion type ordering (matches the i246 PFCT convention so reporting
# / visualisation columns line up).
PROMOTION_TYPE_NAMES: tuple[str, ...] = ("Q", "R", "B", "N")
PROMOTION_TYPE_COUNT = len(PROMOTION_TYPE_NAMES)

# Per-promotion-type directional subset (indices into DIRECTIONS). Queen
# uses all 8; rook uses the 4 orthogonals; bishop uses the 4 diagonals;
# knight has its own template.
QUEEN_DIRECTIONS = tuple(range(NUM_DIRECTIONS))
ROOK_DIRECTIONS = (0, 2, 4, 6)        # N, E, S, W
BISHOP_DIRECTIONS = (1, 3, 5, 7)      # NE, SE, SW, NW

# Knight move offsets in (drow, dfile).
KNIGHT_OFFSETS: tuple[tuple[int, int], ...] = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)

ARRIVAL_FILES = 8
ARRIVAL_SQUARES = ARRIVAL_FILES  # row 0, files 0..7

ALLOWED_ABLATIONS: tuple[str, ...] = (
    "none",
    "pseudo_only",             # falsifier: drop legality filtering on candidates (push always considered legal regardless of occupancy)
    "no_capture",              # falsifier: drop diagonal capture promotion candidates
    "queen_only",              # falsifier: zero out delta-to-queen and knight-fork features
    "no_attack_defense",       # falsifier: zero out arrival-square attack/defense features
    "zero_delta",
    "trunk_only",              # alias for zero_delta
    "disable_gate",
)


def _build_canonicalize_perm() -> torch.Tensor:
    """Per-plane permutation for the black-to-move canonicalisation.

    simple_18 plane layout (see ``board_features.PIECE_PLANES``):

      ``[P, N, B, R, Q, K, p, n, b, r, q, k, stm, WK, WQ, BK, BQ, ep]``

    For black-to-move canonicalisation we swap the white and black
    piece planes (so plane 0 is always "own pawn"), keep the STM plane
    (it will be force-set to 1.0 in canonical space), and swap the
    castling plane pairs ``(WK, WQ) <-> (BK, BQ)`` so the own-side
    castling planes always live at canonical indices 13, 14. The
    en-passant plane is preserved but vertically flipped (the flip is
    applied separately on the whole tensor).
    """
    return torch.tensor(
        [
            6, 7, 8, 9, 10, 11,           # own piece planes (were black)
            0, 1, 2, 3, 4, 5,             # opp piece planes (were white)
            12,                           # STM plane -- value rebuilt to 1
            15, 16,                       # own castling planes (were BK, BQ)
            13, 14,                       # opp castling planes (were WK, WQ)
            17,                           # ep plane (vertical flip handles geometry)
        ],
        dtype=torch.long,
    )


def _build_knight_attack_template() -> torch.Tensor:
    """``(64, 64)`` knight attack template: ``[src, dst] = 1`` iff a knight at src attacks dst."""
    template = np.zeros((SQUARES, SQUARES), dtype=np.float32)
    for src in range(SQUARES):
        sr, sf = src // 8, src % 8
        for dr, df in KNIGHT_OFFSETS:
            r, f = sr + dr, sf + df
            if 0 <= r < 8 and 0 <= f < 8:
                template[src, r * 8 + f] = 1.0
    return torch.from_numpy(template)


def _build_king_zone_template() -> torch.Tensor:
    """``(64, 64)`` king-zone template: ``[src, dst] = 1`` iff dst is within Chebyshev distance 1 of src (inclusive)."""
    template = np.zeros((SQUARES, SQUARES), dtype=np.float32)
    for src in range(SQUARES):
        sr, sf = src // 8, src % 8
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                r, f = sr + dr, sf + df
                if 0 <= r < 8 and 0 <= f < 8:
                    template[src, r * 8 + f] = 1.0
    return torch.from_numpy(template)


def _build_king_attack_template() -> torch.Tensor:
    """``(64, 64)`` king *attack* template (strictly excludes src)."""
    template = np.zeros((SQUARES, SQUARES), dtype=np.float32)
    for src in range(SQUARES):
        sr, sf = src // 8, src % 8
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                r, f = sr + dr, sf + df
                if 0 <= r < 8 and 0 <= f < 8:
                    template[src, r * 8 + f] = 1.0
    return torch.from_numpy(template)


def canonicalize_simple_18(board: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Canonicalise the simple_18 board to side-to-move-up form.

    For white-to-move samples this is the identity. For black-to-move
    samples the board is vertically flipped (rank-axis flip) and the
    piece / castling planes are swapped so the own side's pieces live at
    canonical plane indices 0..5 and the own side's castling rights live
    at indices 13, 14. The STM plane is fixed to 1.0 in canonical space
    so downstream code can read "own to move" directly.

    Args:
        board: ``(B, 18, 8, 8)`` simple_18 tensor.
        perm: ``(18,)`` long tensor produced by :func:`_build_canonicalize_perm`.

    Returns:
        ``(B, 18, 8, 8)`` canonicalised tensor.
    """
    if board.dim() != 4 or board.shape[1] != SIMPLE_18_PLANES or board.shape[-2:] != (8, 8):
        raise ValueError(f"Expected simple_18 tensor (B, 18, 8, 8), got {tuple(board.shape)}")
    batch = board.shape[0]
    stm = (board[:, 12, 0, 0] > 0.5)  # (B,) bool
    flipped = torch.flip(board, dims=[-2])
    swapped = flipped.index_select(dim=1, index=perm.to(board.device))
    selector = stm.view(batch, 1, 1, 1).to(dtype=board.dtype)
    canonical = selector * board + (1.0 - selector) * swapped
    # Force STM plane to a flat "1.0" -- own side always moves in canonical
    # space. Use a clone path to keep the autograd graph clean.
    canonical = canonical.clone()
    canonical[:, 12, :, :] = 1.0
    return canonical


def sliding_attack_masks_from_arrival(
    occupancy: torch.Tensor,
    ray_step_index: torch.Tensor,
    ray_step_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute (B, 8 dirs, 8 src_files, 64) sliding attack mask from the 8 arrival squares.

    For each row-0 source file (treated as a queen-like sliding piece on
    canonical row 0), the attack mask along each direction includes all
    on-board squares up to and including the first blocker, where the
    blocker is read from ``occupancy``. The mask is zero for off-board
    steps and zero for steps strictly past the first blocker. The
    blocker square itself is included (so an enemy piece on the first
    blocker is "attacked").

    Args:
        occupancy: ``(B, 64)`` float occupancy mask (any non-king piece).
        ray_step_index: ``(8, 64, 7)`` long ray index table.
        ray_step_mask: ``(8, 64, 7)`` float on-board mask.
    """
    batch = occupancy.shape[0]
    device = occupancy.device
    dtype = occupancy.dtype

    # Restrict the ray tables to the 8 row-0 source squares.
    ri = ray_step_index[:, :ARRIVAL_SQUARES, :].contiguous()  # (8 dirs, 8 srcs, 7)
    rm = ray_step_mask[:, :ARRIVAL_SQUARES, :].to(device=device, dtype=dtype)

    flat_idx = ri.reshape(-1)  # (8*8*7,)
    gathered = occupancy[:, flat_idx].reshape(batch, NUM_DIRECTIONS, ARRIVAL_SQUARES, RAY_MAX_LEN)
    gathered = gathered * rm.unsqueeze(0)

    # cum_occ_strict_prev[..., l] = max occupancy at steps [0, l-1]
    # (so step l is "blocked-before-l" iff a piece sits at step < l).
    cum_occ_strict_prev = torch.zeros_like(gathered)
    if RAY_MAX_LEN > 1:
        cum_occ_strict_prev[..., 1:] = torch.cummax(gathered[..., :-1], dim=-1).values
    not_blocked_before = (cum_occ_strict_prev < 0.5).to(dtype=dtype)
    attack_at_step = not_blocked_before * rm.unsqueeze(0)  # zero where off-board

    attack_mask = torch.zeros(
        batch,
        NUM_DIRECTIONS,
        ARRIVAL_SQUARES,
        SQUARES,
        device=device,
        dtype=dtype,
    )
    target_idx = ri.unsqueeze(0).expand(batch, NUM_DIRECTIONS, ARRIVAL_SQUARES, RAY_MAX_LEN)
    attack_mask.scatter_add_(dim=-1, index=target_idx, src=attack_at_step)
    attack_mask = attack_mask.clamp(0.0, 1.0)
    return attack_mask  # (B, dirs, srcs, 64)


def first_blocker_pieces_from_arrival(
    occupancy: torch.Tensor,
    piece_type_planes: torch.Tensor,
    ray_step_index: torch.Tensor,
    ray_step_mask: torch.Tensor,
) -> torch.Tensor:
    """For each (direction, arrival square), aggregate the piece-type planes of the first blocker.

    Returns a ``(B, NUM_DIRECTIONS, ARRIVAL_SQUARES, C)`` tensor whose
    last axis sums one-hot piece-type indicators of the first piece met
    along each ray from each arrival square. Zero when no blocker
    exists along that ray.

    Args:
        occupancy: ``(B, 64)`` occupancy mask used to detect blockers.
        piece_type_planes: ``(B, 64, C)`` per-square one-hot of the
            piece types we want to read at the blocker (e.g. 12 colour
            x type planes).
        ray_step_index: ``(8, 64, 7)`` long.
        ray_step_mask: ``(8, 64, 7)`` float.
    """
    batch = occupancy.shape[0]
    channels = piece_type_planes.shape[-1]
    device = occupancy.device
    dtype = occupancy.dtype

    ri = ray_step_index[:, :ARRIVAL_SQUARES, :].contiguous()
    rm = ray_step_mask[:, :ARRIVAL_SQUARES, :].to(device=device, dtype=dtype)

    flat_idx = ri.reshape(-1)
    occ_gathered = occupancy[:, flat_idx].reshape(batch, NUM_DIRECTIONS, ARRIVAL_SQUARES, RAY_MAX_LEN)
    occ_gathered = occ_gathered * rm.unsqueeze(0)

    pieces_gathered = piece_type_planes[:, flat_idx, :].reshape(
        batch, NUM_DIRECTIONS, ARRIVAL_SQUARES, RAY_MAX_LEN, channels
    )
    pieces_gathered = pieces_gathered * rm.unsqueeze(0).unsqueeze(-1)

    # The "first blocker" step l is the smallest l where occupancy at l > 0.
    # Build a per-step indicator: is_first_blocker[..., l] = 1 iff occupancy at l > 0 AND
    # cummax of occupancy at steps [0, l-1] == 0.
    cum_occ_strict_prev = torch.zeros_like(occ_gathered)
    if RAY_MAX_LEN > 1:
        cum_occ_strict_prev[..., 1:] = torch.cummax(occ_gathered[..., :-1], dim=-1).values
    not_blocked_before = (cum_occ_strict_prev < 0.5).to(dtype=dtype)
    blocker_at_step = (occ_gathered > 0.5).to(dtype=dtype) * not_blocked_before
    # At most one step in each (dir, src) row of `blocker_at_step` is 1.

    # Aggregate first-blocker piece types: weight pieces_gathered by blocker_at_step.
    first_pieces = (pieces_gathered * blocker_at_step.unsqueeze(-1)).sum(dim=-2)
    return first_pieces  # (B, dirs, srcs, C)


def compute_arrival_attackers_defenders(
    canonical: torch.Tensor,
    ray_step_index: torch.Tensor,
    ray_step_mask: torch.Tensor,
    knight_template: torch.Tensor,
    king_attack_template: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-arrival-square attacker / defender feature components.

    Returns a dict of ``(B, ARRIVAL_SQUARES)`` tensors:

      - ``opp_attackers_total``: total enemy attackers on each arrival
        square ``u`` from sliding pieces, knights and king.
      - ``own_defenders_total``: total own defenders.

    Sliding attackers are counted by looking at the first blocker along
    each of the 8 ray directions emanating from ``u`` and counting it if
    its piece type matches the ray (rook/queen on orthogonals, bishop/
    queen on diagonals). Knight attackers are counted via the knight
    template (knight attacks are symmetric so "knights attacking u" =
    "knights at knight offsets from u"). King attackers are counted via
    the king-attack template (king at adjacent square -- non-zero only
    when the enemy king happens to be on row 1 file in {f-1, f, f+1}).
    Pawn attackers on an arrival on canonical row 0 are always zero
    (enemy pawns move toward canonical row 7).
    """
    batch = canonical.shape[0]
    device = canonical.device
    dtype = canonical.dtype

    piece_planes = canonical[:, :NUM_PIECE_CHANNELS]  # (B, 12, 8, 8)
    per_square = piece_planes.flatten(2).transpose(1, 2).contiguous()  # (B, 64, 12)
    occupancy_flat = per_square.sum(dim=-1).clamp(0.0, 1.0)  # (B, 64)

    first_pieces = first_blocker_pieces_from_arrival(
        occupancy_flat, per_square, ray_step_index, ray_step_mask
    )  # (B, dirs, srcs, 12)

    # Direction selectors: shape (NUM_DIRECTIONS,) float.
    rook_dir = torch.zeros(NUM_DIRECTIONS, device=device, dtype=dtype)
    for d in ROOK_DIRECTIONS:
        rook_dir[d] = 1.0
    bishop_dir = torch.zeros(NUM_DIRECTIONS, device=device, dtype=dtype)
    for d in BISHOP_DIRECTIONS:
        bishop_dir[d] = 1.0
    queen_dir = torch.ones(NUM_DIRECTIONS, device=device, dtype=dtype)

    # Per-direction first-blocker piece-type signals: extract own / opp views.
    opp_q = first_pieces[..., OPP_QUEEN]   # (B, dirs, srcs)
    opp_r = first_pieces[..., OPP_ROOK]
    opp_b = first_pieces[..., OPP_BISHOP]
    own_q = first_pieces[..., OWN_QUEEN]
    own_r = first_pieces[..., OWN_ROOK]
    own_b = first_pieces[..., OWN_BISHOP]

    # Aggregate sliding attackers from u (= row-0 src files).
    # An enemy queen at the first blocker of any direction attacks u; same for
    # enemy rook on rook directions; enemy bishop on bishop directions.
    opp_sliding = (
        (opp_q * queen_dir.view(1, -1, 1)).sum(dim=1)
        + (opp_r * rook_dir.view(1, -1, 1)).sum(dim=1)
        + (opp_b * bishop_dir.view(1, -1, 1)).sum(dim=1)
    )  # (B, srcs)
    own_sliding = (
        (own_q * queen_dir.view(1, -1, 1)).sum(dim=1)
        + (own_r * rook_dir.view(1, -1, 1)).sum(dim=1)
        + (own_b * bishop_dir.view(1, -1, 1)).sum(dim=1)
    )

    # Knight attackers: enemy knights at knight offsets from u.
    enemy_knights_per_square = per_square[..., OPP_KNIGHT]  # (B, 64)
    own_knights_per_square = per_square[..., OWN_KNIGHT]
    # knight_template[src, dst] = 1 if knight at src attacks dst -- equivalently
    # for our purposes, knight at "dst (= knight-offset of u)" attacks u. We want
    # for each arrival square g on row 0, count of enemy knights at squares
    # `s` such that knight_template[s, g_sq] = 1 == knight_template[g_sq, s] = 1.
    arrival_squares_idx = torch.arange(ARRIVAL_SQUARES, device=device, dtype=torch.long)
    arrival_knight_template = knight_template.to(device=device, dtype=dtype)[arrival_squares_idx]
    # (ARRIVAL_SQUARES, 64)
    opp_knight_attackers = enemy_knights_per_square @ arrival_knight_template.t()
    # (B, ARRIVAL_SQUARES)
    own_knight_defenders = own_knights_per_square @ arrival_knight_template.t()

    # King attackers: enemy king at adjacent square to u.
    enemy_king_per_square = per_square[..., OPP_KING]
    own_king_per_square = per_square[..., OWN_KING]
    arrival_king_attack = king_attack_template.to(device=device, dtype=dtype)[arrival_squares_idx]
    opp_king_attackers = enemy_king_per_square @ arrival_king_attack.t()
    own_king_defenders = own_king_per_square @ arrival_king_attack.t()

    opp_attackers_total = opp_sliding + opp_knight_attackers + opp_king_attackers
    own_defenders_total = own_sliding + own_knight_defenders + own_king_defenders

    return {
        "opp_attackers": opp_attackers_total,
        "own_defenders": own_defenders_total,
        "opp_sliding": opp_sliding,
        "own_sliding": own_sliding,
        "opp_knight_attackers": opp_knight_attackers,
        "own_knight_defenders": own_knight_defenders,
    }


def compute_promoted_attack_features(
    canonical: torch.Tensor,
    ray_step_index: torch.Tensor,
    ray_step_mask: torch.Tensor,
    knight_template: torch.Tensor,
    king_zone_template: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute promoted-piece attack-set features per arrival square and per piece type.

    Returns a dict of ``(B, ARRIVAL_SQUARES)`` tensors per metric per
    piece type, where pieces are indexed in PROMOTION_TYPE_NAMES order
    (Q, R, B, N):

      - ``check[t]``: indicator that the promoted piece of type ``t``
        attacks the enemy king (i.e. it gives check after promotion).
      - ``zone[t]``: count of squares in the enemy king's 3x3 zone
        attacked by the promoted piece of type ``t``.
      - ``hi_value[t]``: weighted sum of enemy high-value targets the
        promoted piece of type ``t`` attacks -- used as the knight-fork
        hint ``kappa_N`` for ``t = N``.
    """
    batch = canonical.shape[0]
    device = canonical.device
    dtype = canonical.dtype

    piece_planes = canonical[:, :NUM_PIECE_CHANNELS]
    per_square = piece_planes.flatten(2).transpose(1, 2).contiguous()  # (B, 64, 12)
    occupancy_flat = per_square.sum(dim=-1).clamp(0.0, 1.0)  # (B, 64)

    # (B, dirs, srcs, 64): mask of squares attacked by a sliding piece on (row 0, src)
    # along each direction, with occlusion.
    slide_attack = sliding_attack_masks_from_arrival(occupancy_flat, ray_step_index, ray_step_mask)

    # Sum directions per piece type.
    queen_attack = slide_attack.sum(dim=1).clamp(0.0, 1.0)  # (B, srcs, 64)
    rook_dir_mask = torch.zeros(NUM_DIRECTIONS, device=device, dtype=dtype)
    bishop_dir_mask = torch.zeros(NUM_DIRECTIONS, device=device, dtype=dtype)
    for d in ROOK_DIRECTIONS:
        rook_dir_mask[d] = 1.0
    for d in BISHOP_DIRECTIONS:
        bishop_dir_mask[d] = 1.0
    rook_attack = (slide_attack * rook_dir_mask.view(1, -1, 1, 1)).sum(dim=1).clamp(0.0, 1.0)
    bishop_attack = (slide_attack * bishop_dir_mask.view(1, -1, 1, 1)).sum(dim=1).clamp(0.0, 1.0)

    # Knight attack from each arrival square (no occlusion).
    arrival_squares_idx = torch.arange(ARRIVAL_SQUARES, device=device, dtype=torch.long)
    knight_attack = knight_template.to(device=device, dtype=dtype)[arrival_squares_idx]
    knight_attack = knight_attack.unsqueeze(0).expand(batch, -1, -1)  # (B, srcs, 64)

    attack_by_type = torch.stack([queen_attack, rook_attack, bishop_attack, knight_attack], dim=1)
    # (B, 4, srcs, 64)

    # Gives-check indicator: does the attack include the enemy king square?
    enemy_king_flat = per_square[..., OPP_KING]  # (B, 64)
    check_per_type = (attack_by_type * enemy_king_flat.view(batch, 1, 1, -1)).sum(dim=-1)
    # (B, 4, srcs)
    check_per_type = check_per_type.clamp(0.0, 1.0)

    # King-zone overlap: count of squares in the enemy king's 3x3 zone attacked.
    enemy_king_zone = enemy_king_flat @ king_zone_template.to(device=device, dtype=dtype)  # (B, 64)
    zone_per_type = (attack_by_type * enemy_king_zone.view(batch, 1, 1, -1)).sum(dim=-1)
    # (B, 4, srcs)

    # High-value targets: enemy Q, R, B, N (weights mirror standard
    # material order). Optionally include enemy king zone with a small
    # weight so a kingside-fork is captured even when the king itself
    # is shielded. The knight-fork hint is taken from this for t=N.
    hi_value_weights = canonical.new_zeros(NUM_PIECE_CHANNELS)
    hi_value_weights[OPP_QUEEN] = 5.0
    hi_value_weights[OPP_ROOK] = 3.0
    hi_value_weights[OPP_BISHOP] = 2.0
    hi_value_weights[OPP_KNIGHT] = 2.0
    hi_value_weights[OPP_KING] = 3.0   # king is captured-equivalent
    hi_value_per_square = per_square @ hi_value_weights  # (B, 64)
    hi_value_per_type = (attack_by_type * hi_value_per_square.view(batch, 1, 1, -1)).sum(dim=-1)

    return {
        "check": check_per_type,        # (B, 4, srcs)
        "zone": zone_per_type,
        "hi_value": hi_value_per_type,
    }


def compute_per_file_candidates(canonical: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute per-file candidate masks and global pawn-distance summaries.

    Returns:
        ``n_own`` and ``n_opp`` global summaries (B, 3) at canonical
        rows 1, 2, 3 (for opp the analogous rows are 6, 5, 4 in
        canonical space). ``push_mask``, ``capL_mask``, ``capR_mask``:
        (B, 8) per-file indicators with values in [0, 1]; the
        ``push_mask`` requires the arrival square (canonical row 0,
        file f) to be empty; the capture masks require the diagonal
        arrival square to be occupied by an enemy piece.
    """
    batch = canonical.shape[0]
    device = canonical.device
    dtype = canonical.dtype

    own_pawn = canonical[:, OWN_PAWN]   # (B, 8, 8)
    opp_pawn = canonical[:, OPP_PAWN]

    # Own near-promotion pawn presence at canonical row 1.
    own_pawn_r1 = (own_pawn[:, 1, :] > 0.5).to(dtype=dtype)  # (B, 8)

    # Global pawn-distance summaries.
    n_own = torch.stack(
        [own_pawn[:, 1, :].sum(dim=-1), own_pawn[:, 2, :].sum(dim=-1), own_pawn[:, 3, :].sum(dim=-1)],
        dim=-1,
    )
    n_opp = torch.stack(
        [opp_pawn[:, 6, :].sum(dim=-1), opp_pawn[:, 5, :].sum(dim=-1), opp_pawn[:, 4, :].sum(dim=-1)],
        dim=-1,
    )

    # Occupancy of canonical row 0.
    own_occ_r0 = canonical[:, :OWN_KING + 1, 0, :].sum(dim=1).clamp(0.0, 1.0)  # (B, 8)
    opp_occ_r0 = canonical[:, OPP_PAWN:OPP_KING + 1, 0, :].sum(dim=1).clamp(0.0, 1.0)
    any_occ_r0 = ((own_occ_r0 + opp_occ_r0) > 0.5).to(dtype=dtype)
    enemy_at_r0 = (opp_occ_r0 > 0.5).to(dtype=dtype)

    # Push candidate: own pawn at (1, f), target (0, f) empty.
    push_mask = own_pawn_r1 * (1.0 - any_occ_r0)

    # Capture left: own pawn at (1, f), target (0, f-1) has enemy.
    enemy_at_left_target = torch.zeros_like(own_pawn_r1)
    enemy_at_left_target[:, 1:] = enemy_at_r0[:, :-1]
    capL_mask = own_pawn_r1 * enemy_at_left_target

    # Capture right: target (0, f+1).
    enemy_at_right_target = torch.zeros_like(own_pawn_r1)
    enemy_at_right_target[:, :-1] = enemy_at_r0[:, 1:]
    capR_mask = own_pawn_r1 * enemy_at_right_target

    return {
        "n_own": n_own,                    # (B, 3)
        "n_opp": n_opp,                    # (B, 3)
        "push_mask": push_mask,            # (B, 8)  per-file (source-file)
        "capL_mask": capL_mask,
        "capR_mask": capR_mask,
        "enemy_at_r0": enemy_at_r0,
        "own_pawn_at_r1": own_pawn_r1,
    }


# Number of per-arrival-square per-candidate features built by
# ``_assemble_per_arrival_features``: see that function for the layout.
PER_ARRIVAL_FEATURE_DIM = 1 + 1 + 3 + (4 * 3) + 1  # mask + edge + (cQ,zQ,sQ) + 4 delta-to-Q (cR-cQ,zR-zQ,sR-sQ,...) + knight_fork
# Total per-candidate token = mask(1) + edge(1) + cQ(1) + zQ(1) + sQ(1) +
# (cR-cQ, zR-zQ, sR-sQ) (3) + (cB-cQ, zB-zQ, sB-sQ) (3) + (cN-cQ, zN-zQ, sN-sQ) (3) +
# kappa_N(1) + capture_flag(1) = 1+1+3+3+3+3+1+1 = 16
# We rebuild this carefully in the function below.


def _assemble_per_arrival_features(
    promoted_attacks: dict[str, torch.Tensor],
    attacker_defender: dict[str, torch.Tensor],
    arrival_to_source_file: torch.Tensor,
    candidate_mask: torch.Tensor,
    is_capture_flag: float,
) -> torch.Tensor:
    """Build per-arrival-square candidate tokens for one candidate-type slice.

    The token layout is::

        [ mask, capture_flag, edge_file,
          cQ, zQ, sQ,
          cR - cQ, zR - zQ, sR - sQ,
          cB - cQ, zB - zQ, sB - sQ,
          cN - cQ, zN - zQ, sN - sQ,
          kappa_N ]

    Args:
        promoted_attacks: output of :func:`compute_promoted_attack_features`.
        attacker_defender: output of :func:`compute_arrival_attackers_defenders`.
        arrival_to_source_file: ``(8,)`` long mapping arrival file to source
            pawn file. For push: identity. For capL: arrival = source - 1.
            For capR: arrival = source + 1.
        candidate_mask: ``(B, 8)`` mask indexed by *arrival file* in [0, 7].
        is_capture_flag: ``1.0`` for capture-promotion candidates, else ``0.0``.
    """
    batch = candidate_mask.shape[0]
    device = candidate_mask.device
    dtype = candidate_mask.dtype

    check = promoted_attacks["check"]      # (B, 4, srcs)
    zone = promoted_attacks["zone"]
    hi_value = promoted_attacks["hi_value"]

    opp_attack = attacker_defender["opp_attackers"]  # (B, srcs)
    own_defend = attacker_defender["own_defenders"]
    safety_raw = own_defend - opp_attack
    safety = safety_raw.clamp(-4.0, 4.0) / 4.0     # (B, srcs)

    # Reorder src axis to mean "arrival square" (since the helpers were
    # already indexed by the arrival square = row-0 file). For capture
    # promotions the arrival_to_source_file map is needed only for
    # diagnostic alignment with the source pawn -- the features above
    # are already arrival-indexed.

    cQ = check[:, 0, :]
    cR = check[:, 1, :]
    cB = check[:, 2, :]
    cN = check[:, 3, :]
    zQ = zone[:, 0, :]
    zR = zone[:, 1, :]
    zB = zone[:, 2, :]
    zN = zone[:, 3, :]

    # Safety scores are piece-agnostic per arrival square (they reflect
    # the balance of attackers/defenders on u, not the promoted piece
    # type). We still expose per-type "safety deltas" via the
    # delta-to-queen channels so the head can learn underpromotion
    # arrival preferences. For piece-specific safety we would need to
    # subtract the source-pawn's own pawn-attacker contribution, which
    # we leave for a future iteration.
    sQ = safety
    sR = safety
    sB = safety
    sN = safety

    edge_file = torch.zeros(ARRIVAL_SQUARES, device=device, dtype=dtype)
    edge_file[0] = 1.0
    edge_file[ARRIVAL_SQUARES - 1] = 1.0
    edge_file = edge_file.unsqueeze(0).expand(batch, -1)  # (B, srcs)

    capture_plane = torch.full_like(edge_file, float(is_capture_flag))

    knight_fork = hi_value[:, 3, :]  # kappa_N per arrival

    tokens = torch.stack(
        [
            candidate_mask,
            capture_plane * candidate_mask,
            edge_file * candidate_mask,
            cQ * candidate_mask,
            zQ * candidate_mask,
            sQ * candidate_mask,
            (cR - cQ) * candidate_mask,
            (zR - zQ) * candidate_mask,
            (sR - sQ) * candidate_mask,
            (cB - cQ) * candidate_mask,
            (zB - zQ) * candidate_mask,
            (sB - sQ) * candidate_mask,
            (cN - cQ) * candidate_mask,
            (zN - zQ) * candidate_mask,
            (sN - sQ) * candidate_mask,
            knight_fork * candidate_mask,
        ],
        dim=-1,
    )
    return tokens  # (B, ARRIVAL_SQUARES, PER_ARRIVAL_FEATURE_DIM)


PER_ARRIVAL_TOKEN_DIM = 16
NUM_CANDIDATE_KINDS = 3  # push, capL, capR


class PromotionUnderpromotionGeometry(nn.Module):
    """p052 -- Promotion and Underpromotion Geometry head over the i193 trunk."""

    def __init__(
        self,
        input_channels: int = 18,
        num_classes: int = 1,
        # Trunk hyper-parameters (i193).
        trunk_channels: int = 64,
        trunk_hidden_dim: int = 96,
        trunk_depth: int = 2,
        trunk_dropout: float = 0.1,
        trunk_use_batchnorm: bool = True,
        trunk_gate_dim: int | None = None,
        trunk_ablation: str = "none",
        # PUGP hyper-parameters.
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        gate_init: float = -2.0,
        ablation: str = "none",
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError(
                "PromotionUnderpromotionGeometry supports the puzzle_binary one-logit contract"
            )
        if int(input_channels) != SIMPLE_18_PLANES:
            raise ValueError(
                "PromotionUnderpromotionGeometry requires the simple_18 board tensor"
            )
        if str(ablation) not in ALLOWED_ABLATIONS:
            raise ValueError(
                f"Unknown ablation={ablation!r}; expected one of {sorted(ALLOWED_ABLATIONS)}"
            )

        self.num_classes = 1
        self.ablation = str(ablation)
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

        ray_idx, ray_mask = build_ray_step_index()
        self.register_buffer("ray_step_index", ray_idx)
        self.register_buffer("ray_step_mask", ray_mask)
        self.register_buffer("knight_template", _build_knight_attack_template())
        self.register_buffer("king_zone_template", _build_king_zone_template())
        self.register_buffer("king_attack_template", _build_king_attack_template())
        self.register_buffer("canonicalize_perm", _build_canonicalize_perm())

        # Pooled feature vector layout:
        #   - n_own (3) + n_opp (3) = 6
        #   - per-candidate-kind (push, capL, capR): sum + max pool of
        #     (PER_ARRIVAL_TOKEN_DIM) tokens over arrival files -> 2 * 16 = 32 each
        #     => 3 kinds * 32 = 96
        #   - candidate counts: |M*_push|, |M*_capL|, |M*_capR|, |M*| = 4
        self.feature_dim_total = (
            6
            + NUM_CANDIDATE_KINDS * 2 * PER_ARRIVAL_TOKEN_DIM
            + 4
        )

        self.feature_norm = nn.LayerNorm(self.feature_dim_total)
        self.trunk_feature_dim = (
            2 * self.trunk.exchange_encoder.output_dim
            + DualStreamFeatureBuilder.SUMMARY_DIM
        )
        dropout_module: nn.Module = (
            nn.Dropout(float(head_dropout)) if float(head_dropout) > 0 else nn.Identity()
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim_total + self.trunk_feature_dim),
            nn.Linear(self.feature_dim_total + self.trunk_feature_dim, int(head_hidden_dim)),
            nn.GELU(),
            dropout_module,
            nn.Linear(int(head_hidden_dim), 1),
        )
        gate_extra_dim = 4  # candidate_total, n_own_r1, n_opp_r1, has_capture
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.trunk_feature_dim + gate_extra_dim),
            nn.Linear(self.trunk_feature_dim + gate_extra_dim, max(16, int(head_hidden_dim) // 2)),
            nn.GELU(),
            nn.Linear(max(16, int(head_hidden_dim) // 2), 1),
        )
        with torch.no_grad():
            final = self.gate_head[-1]
            if isinstance(final, nn.Linear):
                final.bias.fill_(float(gate_init))

    def _candidate_masks_with_ablation(
        self,
        candidates: dict[str, torch.Tensor],
        canonical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        push = candidates["push_mask"]
        capL = candidates["capL_mask"]
        capR = candidates["capR_mask"]

        if self.ablation == "pseudo_only":
            # Drop legality filtering: every own near-promotion pawn yields a
            # quiet push candidate regardless of occupancy on the arrival
            # square, and capture candidates fire whenever the diagonal
            # arrival square exists (regardless of whether it actually has
            # an enemy piece).
            own_r1 = candidates["own_pawn_at_r1"]
            push = own_r1
            capL = torch.zeros_like(own_r1)
            capR = torch.zeros_like(own_r1)
            capL[:, 1:] = own_r1[:, 1:]
            capR[:, :-1] = own_r1[:, :-1]
        elif self.ablation == "no_capture":
            capL = torch.zeros_like(capL)
            capR = torch.zeros_like(capR)

        # Re-key by *arrival file* (the source-file mask is what the helper
        # returns; capL arrival file = source - 1, capR arrival file = source + 1).
        push_arrival = push
        capL_arrival = torch.zeros_like(capL)
        capL_arrival[:, :-1] = capL[:, 1:]
        capR_arrival = torch.zeros_like(capR)
        capR_arrival[:, 1:] = capR[:, :-1]

        return {
            "push": push_arrival,
            "capL": capL_arrival,
            "capR": capR_arrival,
        }

    def _build_feature_vector(self, canonical: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = canonical.shape[0]
        device = canonical.device
        dtype = canonical.dtype

        candidates = compute_per_file_candidates(canonical)
        attacker_defender = compute_arrival_attackers_defenders(
            canonical,
            self.ray_step_index,
            self.ray_step_mask,
            self.knight_template,
            self.king_attack_template,
        )
        promoted_attacks = compute_promoted_attack_features(
            canonical,
            self.ray_step_index,
            self.ray_step_mask,
            self.knight_template,
            self.king_zone_template,
        )

        arrival_masks = self._candidate_masks_with_ablation(candidates, canonical)
        push_mask = arrival_masks["push"]
        capL_mask = arrival_masks["capL"]
        capR_mask = arrival_masks["capR"]

        if self.ablation == "queen_only":
            # Zero out delta-to-queen columns by zeroing R/B/N check/zone
            # signals before assembly: easiest via passing modified
            # promoted_attacks.
            zeroed = {k: v.clone() for k, v in promoted_attacks.items()}
            for ch in ("check", "zone"):
                zeroed[ch][:, 1:, :] = zeroed[ch][:, 0:1, :]
            zeroed["hi_value"] = torch.zeros_like(zeroed["hi_value"])
            promoted_attacks = zeroed
        if self.ablation == "no_attack_defense":
            ad_zero = {k: torch.zeros_like(v) for k, v in attacker_defender.items()}
            attacker_defender = ad_zero
            zeroed = {k: v.clone() for k, v in promoted_attacks.items()}
            for ch in ("check", "zone"):
                zeroed[ch][:] = 0.0
            zeroed["hi_value"] = torch.zeros_like(zeroed["hi_value"])
            promoted_attacks = zeroed

        # Arrival-to-source maps (kept for diagnostics; tokens are
        # arrival-indexed throughout).
        identity_files = torch.arange(ARRIVAL_SQUARES, device=device, dtype=torch.long)

        push_tokens = _assemble_per_arrival_features(
            promoted_attacks,
            attacker_defender,
            identity_files,
            push_mask,
            is_capture_flag=0.0,
        )
        capL_tokens = _assemble_per_arrival_features(
            promoted_attacks,
            attacker_defender,
            identity_files,
            capL_mask,
            is_capture_flag=1.0,
        )
        capR_tokens = _assemble_per_arrival_features(
            promoted_attacks,
            attacker_defender,
            identity_files,
            capR_mask,
            is_capture_flag=1.0,
        )

        # Pool over arrival files: sum + max per candidate kind.
        def pool(tokens: torch.Tensor) -> torch.Tensor:
            sum_pool = tokens.sum(dim=1)
            max_pool = tokens.max(dim=1).values
            return torch.cat([sum_pool, max_pool], dim=-1)

        push_pool = pool(push_tokens)
        capL_pool = pool(capL_tokens)
        capR_pool = pool(capR_tokens)

        push_count = push_mask.sum(dim=-1, keepdim=True)
        capL_count = capL_mask.sum(dim=-1, keepdim=True)
        capR_count = capR_mask.sum(dim=-1, keepdim=True)
        total_count = push_count + capL_count + capR_count

        feature_vec = torch.cat(
            [
                candidates["n_own"],
                candidates["n_opp"],
                push_pool,
                capL_pool,
                capR_pool,
                push_count,
                capL_count,
                capR_count,
                total_count,
            ],
            dim=-1,
        )

        diagnostics = {
            "push_count": push_count.squeeze(-1),
            "capL_count": capL_count.squeeze(-1),
            "capR_count": capR_count.squeeze(-1),
            "total_count": total_count.squeeze(-1),
            "n_own_r1": candidates["n_own"][:, 0],
            "n_opp_r1": candidates["n_opp"][:, 0],
            "knight_fork_max": promoted_attacks["hi_value"][:, 3, :].max(dim=-1).values,
            "queen_check_count": (
                promoted_attacks["check"][:, 0, :] * (push_mask + capL_mask + capR_mask).clamp(0.0, 1.0)
            ).sum(dim=-1),
            "queen_zone_max": (
                promoted_attacks["zone"][:, 0, :] * (push_mask + capL_mask + capR_mask).clamp(0.0, 1.0)
            ).max(dim=-1).values,
        }
        return feature_vec, diagnostics

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        board = require_board_tensor(x, self.spec)
        batch = board.shape[0]

        trunk_out = self.trunk(board)
        base_logit = trunk_out["logits"].view(-1)
        joint, _, _ = trunk_joint_features(self.trunk, board)

        canonical = canonicalize_simple_18(board, self.canonicalize_perm)
        feature_vec, diagnostics = self._build_feature_vector(canonical)

        feature_norm = self.feature_norm(feature_vec)
        delta_input = torch.cat([feature_norm, joint], dim=-1)
        delta_raw = self.delta_head(delta_input).view(-1)

        has_capture = ((diagnostics["capL_count"] + diagnostics["capR_count"]) > 0.5).to(joint.dtype)
        gate_extras = torch.stack(
            [
                diagnostics["total_count"],
                diagnostics["n_own_r1"],
                diagnostics["n_opp_r1"],
                has_capture,
            ],
            dim=-1,
        )
        gate_input = torch.cat([joint, gate_extras], dim=-1)
        gate_logit = self.gate_head(gate_input).view(-1)
        gate = torch.sigmoid(gate_logit)
        if self.ablation == "disable_gate":
            gate = torch.ones_like(gate)

        if self.ablation in {"zero_delta", "trunk_only"}:
            primitive_delta = torch.zeros_like(base_logit)
            gate_applied = torch.zeros_like(gate)
        else:
            primitive_delta = delta_raw
            gate_applied = gate
        contribution = gate_applied * primitive_delta
        logits = base_logit + contribution

        eps = 1.0e-6
        gate_clamped = gate.clamp(eps, 1.0 - eps)
        gate_entropy = -(
            gate_clamped * gate_clamped.log()
            + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()
        )

        out: dict[str, torch.Tensor] = {}
        for key, value in trunk_out.items():
            if key in {"logits", "proposal_profile_strength", "proposal_keyword_count"}:
                continue
            out[f"trunk_{key}"] = value
        out["logits"] = logits
        out["base_logit"] = base_logit
        out["primitive_delta"] = primitive_delta
        out["primitive_delta_raw"] = delta_raw
        out["primitive_gate"] = gate
        out["primitive_gate_applied"] = gate_applied
        out["primitive_gate_logit"] = gate_logit
        out["primitive_gate_entropy"] = gate_entropy
        out["primitive_contribution"] = contribution
        out["pugp_push_count"] = diagnostics["push_count"]
        out["pugp_capL_count"] = diagnostics["capL_count"]
        out["pugp_capR_count"] = diagnostics["capR_count"]
        out["pugp_total_count"] = diagnostics["total_count"]
        out["pugp_n_own_r1"] = diagnostics["n_own_r1"]
        out["pugp_n_opp_r1"] = diagnostics["n_opp_r1"]
        out["pugp_knight_fork_max"] = diagnostics["knight_fork_max"]
        out["pugp_queen_check_count"] = diagnostics["queen_check_count"]
        out["pugp_queen_zone_max"] = diagnostics["queen_zone_max"]
        out["mechanism_energy"] = trunk_out["mechanism_energy"] + diagnostics["total_count"].detach()
        out["proposal_profile_strength"] = primitive_delta.detach().abs() * gate_entropy
        out["proposal_keyword_count"] = logits.new_full(
            (batch,), float(self.feature_dim_total)
        )
        return out


def build_promotion_underpromotion_from_config(
    config: dict[str, Any],
) -> PromotionUnderpromotionGeometry:
    cfg = dict(config)
    return PromotionUnderpromotionGeometry(
        input_channels=int(cfg.get("input_channels", 18)),
        num_classes=int(cfg.get("num_classes", 1)),
        trunk_channels=int(cfg.get("trunk_channels", cfg.get("channels", 64))),
        trunk_hidden_dim=int(cfg.get("trunk_hidden_dim", cfg.get("hidden_dim", 96))),
        trunk_depth=int(cfg.get("trunk_depth", cfg.get("depth", 2))),
        trunk_dropout=float(cfg.get("trunk_dropout", cfg.get("dropout", 0.1))),
        trunk_use_batchnorm=bool(cfg.get("trunk_use_batchnorm", cfg.get("use_batchnorm", True))),
        trunk_gate_dim=cfg.get("trunk_gate_dim", cfg.get("gate_dim")),
        trunk_ablation=str(cfg.get("trunk_ablation", "none")),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 64)),
        head_dropout=float(cfg.get("head_dropout", 0.1)),
        gate_init=float(cfg.get("gate_init", -2.0)),
        ablation=str(cfg.get("ablation", "none")),
    )


__all__ = (
    "ALLOWED_ABLATIONS",
    "PROMOTION_TYPE_NAMES",
    "PER_ARRIVAL_TOKEN_DIM",
    "NUM_CANDIDATE_KINDS",
    "PromotionUnderpromotionGeometry",
    "build_promotion_underpromotion_from_config",
    "canonicalize_simple_18",
    "compute_arrival_attackers_defenders",
    "compute_per_file_candidates",
    "compute_promoted_attack_features",
    "first_blocker_pieces_from_arrival",
    "sliding_attack_masks_from_arrival",
)
