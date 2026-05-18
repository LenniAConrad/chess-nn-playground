"""Candidate Move Forcedness Sheaf model for idea i251.

i251 keeps i018's exact side-to-move-oriented tactical sheaf trunk unchanged
and wraps it with a small, deterministic candidate-move bottleneck that
explicitly asks: does this position contain one move whose local structural
evidence sharply dominates the alternatives?

The trunk emits the i018 base logit and per-square states. A bounded set of
candidate moves is enumerated deterministically from the canonical
`piece_state` and `occupancy`, scored by a shared per-move encoder that reads
from source/target square states, deterministic chess features (check,
promotion, pin, king-zone, capture flags) and a tiny local sheaf summary, and
then pooled with a top-k softmax bottleneck. A gated additive delta head
augments the i018 logit with a forcedness-aware residual.

The delta head and gate are zero-initialized so the network starts as i018 at
init and only deviates once the move branch finds real signal. The forward
emits the standard i018 diagnostic bundle plus eleven candidate-move
diagnostics for audit and slice analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from chess_nn_playground.models.trunk.idea_blocks import require_board_tensor
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    RELATION_NAMES,
    BoardState,
    OrientedTacticalSheafNet,
    TacticalIncidence,
    _format_logits,
    _weighted_mean,
)


# Structural move kinds. Order matters: the index doubles as a tactical flag
# offset for promotion / underpromotion in the diagnostic readout.
MOVE_KIND_NAMES: tuple[str, ...] = (
    "quiet",
    "capture",
    "promo_q",
    "promo_r",
    "promo_b",
    "promo_n",
)
MOVE_KIND_COUNT: int = len(MOVE_KIND_NAMES)

# Number of deterministic per-move flags emitted by `CandidateMoveBuilder`:
# gives_check, capture, source_pinned, pin_aligned, enters_their_king_zone,
# target_defended_raw, target_defended_unpinned, promotion, underpromotion.
MOVE_FLAG_COUNT: int = 9


def _coords() -> tuple[torch.Tensor, torch.Tensor]:
    rank = torch.arange(64, dtype=torch.long) // 8
    file = torch.arange(64, dtype=torch.long) % 8
    return rank, file


def _knight_offsets() -> tuple[tuple[int, int], ...]:
    return (
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2),
    )


def _ray_step_table() -> torch.Tensor:
    """For every (src, dst) on an aligned ray, return the unit step (dr, df).

    Shape `(64, 64, 2)`. Squares that are not on a rook or bishop ray from
    `src` to `dst` get `(0, 0)`. The same square pair gets `(0, 0)` as well.
    """
    rank, file = _coords()
    dr = rank.view(64, 1) - rank.view(1, 64)
    df = file.view(64, 1) - file.view(1, 64)
    # Note: dr is src_rank - dst_rank. We want step from src to dst, so flip.
    step_r = -dr.sign()
    step_f = -df.sign()
    abs_dr = dr.abs()
    abs_df = df.abs()
    aligned = (abs_dr == 0) | (abs_df == 0) | (abs_dr == abs_df)
    not_self = (abs_dr + abs_df) > 0
    valid = aligned & not_self
    step = torch.stack(
        [
            torch.where(valid, step_r, torch.zeros_like(step_r)),
            torch.where(valid, step_f, torch.zeros_like(step_f)),
        ],
        dim=-1,
    )
    return step


def _knight_neighbours() -> torch.Tensor:
    """For every square, return the up-to-eight knight-target indices padded to 8.

    Shape `(64, 8)` of dtype `long`. Out-of-board entries are filled with
    `-1` so the caller can mask them.
    """
    rank, file = _coords()
    out = torch.full((64, 8), -1, dtype=torch.long)
    for src in range(64):
        sr = int(rank[src])
        sf = int(file[src])
        slot = 0
        for dr, df in _knight_offsets():
            tr = sr + dr
            tf = sf + df
            if 0 <= tr < 8 and 0 <= tf < 8:
                out[src, slot] = tr * 8 + tf
                slot += 1
    return out


def _king_neighbours() -> torch.Tensor:
    """For every square, up-to-eight king-step target indices padded to 8."""
    rank, file = _coords()
    out = torch.full((64, 8), -1, dtype=torch.long)
    for src in range(64):
        sr = int(rank[src])
        sf = int(file[src])
        slot = 0
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                tr = sr + dr
                tf = sf + df
                if 0 <= tr < 8 and 0 <= tf < 8:
                    out[src, slot] = tr * 8 + tf
                    slot += 1
    return out


def _pawn_forward_targets() -> torch.Tensor:
    """Mover-oriented (always moving from low rank to high rank) pawn targets.

    Returns `(64, 4)` indices in order
    `[single_push, double_push, capture_left, capture_right]`. Out-of-board
    targets and pawn-not-on-source are filled with `-1`.
    """
    rank, file = _coords()
    out = torch.full((64, 4), -1, dtype=torch.long)
    for src in range(64):
        sr = int(rank[src])
        sf = int(file[src])
        if sr == 0 or sr >= 7:
            continue
        out[src, 0] = (sr + 1) * 8 + sf
        if sr == 1:
            out[src, 1] = (sr + 2) * 8 + sf
        if sf - 1 >= 0:
            out[src, 2] = (sr + 1) * 8 + (sf - 1)
        if sf + 1 < 8:
            out[src, 3] = (sr + 1) * 8 + (sf + 1)
    return out


@dataclass(frozen=True)
class CandidateMoves:
    """Bounded deterministic candidate-move table.

    All tensors have leading batch dimension `B`. `mask` is 1 on real moves
    and 0 on padding slots. Each candidate is identified by a structural
    `kind` index (see ``MOVE_KIND_NAMES``).
    """

    source: torch.Tensor  # (B, K) long
    target: torch.Tensor  # (B, K) long
    kind: torch.Tensor    # (B, K) long
    mask: torch.Tensor    # (B, K) float
    flags: torch.Tensor   # (B, K, MOVE_FLAG_COUNT) float
    kind_onehot: torch.Tensor  # (B, K, MOVE_KIND_COUNT) float


class CandidateMoveBuilder(nn.Module):
    """Deterministic pseudo-legal candidate-move builder.

    Reuses the canonical mover-oriented `piece_state` and `occupancy` from
    `BoardStateAdapter`. Produces a bounded `(B, K)` candidate table for a
    fixed `max_candidates`. The construction is pseudo-legal in the sense
    of the i251 research packet: knight and king single-step moves, sliding
    moves limited by the deterministic visibility used in i018's ray
    builders, pawn pushes (single + double from rank 1) and pawn captures
    (only on occupied enemy squares). No castling, no en-passant, and no
    king-safety filter in the default mode.

    Padding slots have ``mask=0`` and carry an arbitrary source/target index
    (the source square) so they remain safe under gather operations.
    """

    def __init__(
        self,
        max_candidates: int = 96,
        include_promotions: bool = True,
        promotion_kinds: tuple[str, ...] = ("promo_q", "promo_n"),
    ) -> None:
        super().__init__()
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self.max_candidates = int(max_candidates)
        self.include_promotions = bool(include_promotions)
        promo_indices = []
        for name in promotion_kinds:
            if name not in MOVE_KIND_NAMES:
                raise ValueError(f"unknown promotion kind {name!r}")
            promo_indices.append(MOVE_KIND_NAMES.index(name))
        self.promotion_kind_indices = tuple(sorted(set(promo_indices)))

        self.register_buffer("knight_targets", _knight_neighbours(), persistent=False)
        self.register_buffer("king_targets", _king_neighbours(), persistent=False)
        self.register_buffer("pawn_targets", _pawn_forward_targets(), persistent=False)
        self.register_buffer("ray_step", _ray_step_table(), persistent=False)

    def forward(
        self,
        piece_state: torch.Tensor,
        occupancy: torch.Tensor,
        between: torch.Tensor,
        knight_geometry: torch.Tensor,
        king_geometry: torch.Tensor,
        rook_geometry: torch.Tensor,
        bishop_geometry: torch.Tensor,
        our_attack: torch.Tensor,
        them_attack: torch.Tensor,
        pin_mask: torch.Tensor,
    ) -> CandidateMoves:
        batch = piece_state.shape[0]
        device = piece_state.device
        dtype = piece_state.dtype
        empty = piece_state[:, :, 0].clamp(0.0, 1.0)
        our_piece = piece_state[:, :, 1:7].sum(dim=-1).clamp(0.0, 1.0)
        them_piece = piece_state[:, :, 7:13].sum(dim=-1).clamp(0.0, 1.0)

        # Per-piece presence indicators on the mover side.
        our_pawn = piece_state[:, :, 1].clamp(0.0, 1.0)
        our_knight = piece_state[:, :, 2].clamp(0.0, 1.0)
        our_bishop = piece_state[:, :, 3].clamp(0.0, 1.0)
        our_rook = piece_state[:, :, 4].clamp(0.0, 1.0)
        our_queen = piece_state[:, :, 5].clamp(0.0, 1.0)
        our_king = piece_state[:, :, 6].clamp(0.0, 1.0)

        # "between empty" check for sliders: an edge (s -> t) is open if
        # every square strictly between s and t is empty. We compute
        # `clear[b, s, t] = product over q of (1 - occupancy[b, q])^{between[s,t,q]}`
        # via einsum on log space; since occupancy is in {0, 1} we can use a
        # simpler subtraction: `blockers = einsum("stq,bq->bst", between, occ)`
        # then `clear = 1 - clamp(blockers, 0, 1)`.
        blockers = torch.einsum("stq,bq->bst", between, occupancy)
        clear = (1.0 - blockers.clamp(0.0, 1.0))
        rook_clear = rook_geometry.unsqueeze(0) * clear
        bishop_clear = bishop_geometry.unsqueeze(0) * clear
        # Destination must be either empty or an enemy piece (no friendly capture).
        dst_safe = (empty + them_piece).clamp(0.0, 1.0)
        # Per-square edge availability for each piece kind:
        knight_edge = (
            knight_geometry.unsqueeze(0) * dst_safe.unsqueeze(1) * our_knight.unsqueeze(-1)
        )
        king_edge = (
            king_geometry.unsqueeze(0) * dst_safe.unsqueeze(1) * our_king.unsqueeze(-1)
        )
        rook_edge = rook_clear * dst_safe.unsqueeze(1) * our_rook.unsqueeze(-1)
        bishop_edge = bishop_clear * dst_safe.unsqueeze(1) * our_bishop.unsqueeze(-1)
        queen_edge = (
            (rook_clear + bishop_clear).clamp(0.0, 1.0)
            * dst_safe.unsqueeze(1)
            * our_queen.unsqueeze(-1)
        )

        # Pawn moves: pushes only onto empty squares, captures only onto enemy squares.
        # Mover-oriented forward direction is from low rank to high rank.
        pawn_single = torch.zeros(batch, 64, 64, device=device, dtype=dtype)
        pawn_double = torch.zeros(batch, 64, 64, device=device, dtype=dtype)
        pawn_capture_left = torch.zeros(batch, 64, 64, device=device, dtype=dtype)
        pawn_capture_right = torch.zeros(batch, 64, 64, device=device, dtype=dtype)
        pawn_targets = self.pawn_targets.to(device)
        sources = torch.arange(64, device=device)
        # Single push: target = pawn_targets[:, 0], requires empty target.
        single_dst = pawn_targets[:, 0]
        valid_single = single_dst >= 0
        if valid_single.any():
            src_idx = sources[valid_single]
            dst_idx = single_dst[valid_single]
            pawn_single[:, src_idx, dst_idx] = (
                our_pawn[:, src_idx] * empty[:, dst_idx]
            )
        # Double push: target = pawn_targets[:, 1] from rank 1, requires both
        # the intermediate (single push square) and target square to be empty.
        double_dst = pawn_targets[:, 1]
        valid_double = double_dst >= 0
        if valid_double.any():
            src_idx = sources[valid_double]
            dst_idx = double_dst[valid_double]
            mid_idx = pawn_targets[valid_double, 0]
            pawn_double[:, src_idx, dst_idx] = (
                our_pawn[:, src_idx] * empty[:, mid_idx] * empty[:, dst_idx]
            )
        # Diagonal captures: enemy must occupy the target.
        capture_left = pawn_targets[:, 2]
        valid_left = capture_left >= 0
        if valid_left.any():
            src_idx = sources[valid_left]
            dst_idx = capture_left[valid_left]
            pawn_capture_left[:, src_idx, dst_idx] = (
                our_pawn[:, src_idx] * them_piece[:, dst_idx]
            )
        capture_right = pawn_targets[:, 3]
        valid_right = capture_right >= 0
        if valid_right.any():
            src_idx = sources[valid_right]
            dst_idx = capture_right[valid_right]
            pawn_capture_right[:, src_idx, dst_idx] = (
                our_pawn[:, src_idx] * them_piece[:, dst_idx]
            )
        pawn_edge = (
            pawn_single + pawn_double + pawn_capture_left + pawn_capture_right
        ).clamp(0.0, 1.0)

        edge_sum = (knight_edge + king_edge + rook_edge + bishop_edge + queen_edge + pawn_edge).clamp(0.0, 1.0)

        # Build scores per (src, dst) used to pick the top moves. Scoring
        # favours captures and king-zone entries so the bounded budget is
        # used for tactically relevant edges; padding slots get -inf scores.
        capture_flag = (edge_sum * them_piece.unsqueeze(1)).clamp(0.0, 1.0)
        score = (
            edge_sum
            + 2.0 * capture_flag
            + 0.5 * (edge_sum * our_attack)
            + 0.5 * (edge_sum * them_attack)
        )
        # Mask away padding (no edge).
        score = torch.where(edge_sum > 0.0, score, edge_sum.new_full(edge_sum.shape, float("-inf")))
        flat_score = score.reshape(batch, -1)
        k = min(self.max_candidates, flat_score.shape[1])
        topk = torch.topk(flat_score, k=k, dim=-1)
        flat_indices = topk.indices  # (B, K)
        flat_values = topk.values

        src_idx = (flat_indices // 64)
        dst_idx = (flat_indices % 64)
        mask = (flat_values > float("-inf")).to(dtype)
        # Pad to exactly max_candidates if k < max_candidates (happens only
        # when 64*64 < max_candidates which we never set).
        if k < self.max_candidates:
            pad = self.max_candidates - k
            src_idx = torch.cat([src_idx, src_idx.new_zeros(batch, pad)], dim=1)
            dst_idx = torch.cat([dst_idx, dst_idx.new_zeros(batch, pad)], dim=1)
            mask = torch.cat([mask, mask.new_zeros(batch, pad)], dim=1)
        # Replace any -inf-padded sources with a safe index (square 0).
        # mask=0 ensures these never contribute to the pooled forcedness.
        src_idx = torch.where(mask > 0, src_idx, torch.zeros_like(src_idx))
        dst_idx = torch.where(mask > 0, dst_idx, torch.zeros_like(dst_idx))

        # Per-move structural kind: pick the dominant edge type at the
        # selected (src, dst). Promotion is tagged from the canonical rank
        # of the destination plus the moving piece being a pawn.
        gather_idx = (src_idx * 64 + dst_idx)
        flat_pawn = pawn_edge.reshape(batch, -1)
        flat_capture = capture_flag.reshape(batch, -1)
        is_pawn_move = torch.gather(flat_pawn, 1, gather_idx)
        is_capture = torch.gather(flat_capture, 1, gather_idx)
        # Mover-oriented promotion rank for pawns is the back rank (rank 7).
        dst_rank = dst_idx // 8
        is_promotion = ((dst_rank == 7) & (is_pawn_move > 0)).to(dtype)

        kind = torch.full((batch, self.max_candidates), MOVE_KIND_NAMES.index("quiet"), device=device, dtype=torch.long)
        kind = torch.where(is_capture > 0, torch.full_like(kind, MOVE_KIND_NAMES.index("capture")), kind)
        if self.include_promotions:
            # Promote both as queen and knight if requested; we expand by
            # marking the selected pawn move as one kind based on capture vs not.
            promo_q = torch.full_like(kind, MOVE_KIND_NAMES.index("promo_q"))
            promo_n = torch.full_like(kind, MOVE_KIND_NAMES.index("promo_n"))
            kind = torch.where(is_promotion > 0, promo_q, kind)
            # For an underpromotion proxy, tag every pawn capture that
            # lands on the back rank as a `promo_n` candidate alongside
            # `promo_q`. We do not duplicate the slot; the diagnostic
            # `underpromotion_mass` counts pawn-promotion captures.
            del promo_n
        kind_onehot = torch.nn.functional.one_hot(kind, num_classes=MOVE_KIND_COUNT).to(dtype)

        # Tactical flags computed for selected (src, dst):
        flat_our_attack_target = our_attack.sum(dim=1).reshape(batch, 64)  # B x 64 of edges into square
        flat_them_attack_target = them_attack.sum(dim=1).reshape(batch, 64)
        target_defended_raw = torch.gather(flat_our_attack_target.clamp(0.0, 1.0), 1, dst_idx) * (is_capture > 0).to(dtype)
        target_defended_unpinned = (
            target_defended_raw
            * (1.0 - torch.gather(pin_mask.sum(dim=1).clamp(0.0, 1.0), 1, dst_idx))
        )
        # Source pinned: any pin line includes the source.
        flat_pin_in = pin_mask.sum(dim=1).clamp(0.0, 1.0)  # destinations side
        flat_pin_out = pin_mask.sum(dim=2).clamp(0.0, 1.0)  # source side
        source_pinned = torch.gather(flat_pin_out, 1, src_idx)
        # Pin-aligned move: pin line includes both source and target.
        edge_pin = pin_mask  # (B, 64, 64) where source = attacker dim, target = blocker dim
        pin_at_edge = torch.gather(
            edge_pin.reshape(batch, -1), 1, gather_idx
        )
        pin_aligned = ((source_pinned > 0) & (pin_at_edge > 0)).to(dtype)

        # Enters their king zone: approximate by checking whether the target
        # belongs to the king-zone attack field already built for the trunk.
        king_zone_them_targets = (
            (our_attack * empty.unsqueeze(1)).sum(dim=1).clamp(0.0, 1.0)
        )
        enters_king_zone = torch.gather(king_zone_them_targets, 1, dst_idx)

        # gives_check proxy: the destination is one of their king's neighbours
        # of the moving piece kind. We approximate by detecting whether the
        # target square attacks the enemy king under the same piece geometry.
        # Simpler proxy: the moving piece type's attack pattern includes the
        # enemy king's square once placed on the target. To avoid recomputing
        # full attacks, we use our_attack at the destination as a proxy for
        # "after the move, attacks the enemy king" — strictly an upper bound
        # but cheap and deterministic.
        them_king = piece_state[:, :, 12].clamp(0.0, 1.0)
        king_attack_score = torch.einsum("bst,bt->bs", our_attack, them_king).clamp(0.0, 1.0)
        gives_check = torch.gather(king_attack_score, 1, dst_idx) * mask

        flags = torch.stack(
            [
                gives_check,
                is_capture,
                source_pinned,
                pin_aligned,
                enters_king_zone,
                target_defended_raw,
                target_defended_unpinned,
                is_promotion,
                # underpromotion proxy: pawn capture onto back rank.
                (is_promotion * is_capture),
            ],
            dim=-1,
        )

        return CandidateMoves(
            source=src_idx,
            target=dst_idx,
            kind=kind,
            mask=mask,
            flags=flags * mask.unsqueeze(-1),
            kind_onehot=kind_onehot * mask.unsqueeze(-1),
        )


class MoveLocalSheafSummary(nn.Module):
    """Per-move local sheaf summary read off the trunk's relation masks.

    For each candidate move `(s, t)`, accumulates the per-relation in/out
    edge counts at the source and the target. Outputs a fixed-width
    `(B, K, 4 * R)` summary, where the four blocks are
    `[src_in, src_out, dst_in, dst_out]` per relation. This gives the move
    encoder a direct read of i018's local tactical pressure at the squares
    touched by the move without recomputing anything.
    """

    def __init__(self, relation_count: int) -> None:
        super().__init__()
        self.relation_count = int(relation_count)
        self.output_dim = 4 * int(relation_count)

    def forward(
        self,
        relation_masks: torch.Tensor,  # (B, R, 64, 64)
        candidates: CandidateMoves,
    ) -> torch.Tensor:
        batch, relations, squares, _ = relation_masks.shape
        in_degree = relation_masks.sum(dim=2)   # (B, R, 64)
        out_degree = relation_masks.sum(dim=3)  # (B, R, 64)
        src = candidates.source  # (B, K)
        dst = candidates.target  # (B, K)
        # Gather: index per (batch, relation, K) by source / dst.
        src_expand = src.unsqueeze(1).expand(batch, relations, src.shape[1])
        dst_expand = dst.unsqueeze(1).expand(batch, relations, dst.shape[1])
        src_in = torch.gather(in_degree, 2, src_expand).transpose(1, 2)
        src_out = torch.gather(out_degree, 2, src_expand).transpose(1, 2)
        dst_in = torch.gather(in_degree, 2, dst_expand).transpose(1, 2)
        dst_out = torch.gather(out_degree, 2, dst_expand).transpose(1, 2)
        summary = torch.cat([src_in, src_out, dst_in, dst_out], dim=-1) / 8.0
        return summary * candidates.mask.unsqueeze(-1)


class CandidateMoveEncoder(nn.Module):
    """Per-move encoder: square states + flags + kind onehot + sheaf summary."""

    def __init__(
        self,
        d_model: int,
        sheaf_summary_dim: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.embed_dim = int(embed_dim)
        input_dim = (
            3 * d_model
            + MOVE_FLAG_COUNT
            + MOVE_KIND_COUNT
            + sheaf_summary_dim
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.score_head = nn.Linear(embed_dim, 1)
        # Zero-init the score head so the top-k pooling produces uniform
        # weights at init -- this keeps the move branch ineffective until
        # training has a reason to differentiate moves.
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def forward(
        self,
        h: torch.Tensor,                # (B, 64, d_model)
        candidates: CandidateMoves,
        sheaf_summary: torch.Tensor,    # (B, K, sheaf_summary_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, squares, d = h.shape
        k = candidates.source.shape[1]
        src_h = torch.gather(h, 1, candidates.source.unsqueeze(-1).expand(batch, k, d))
        dst_h = torch.gather(h, 1, candidates.target.unsqueeze(-1).expand(batch, k, d))
        diff = dst_h - src_h
        fused = torch.cat(
            [
                src_h,
                dst_h,
                diff,
                candidates.flags,
                candidates.kind_onehot,
                sheaf_summary,
            ],
            dim=-1,
        )
        embed = self.mlp(fused)
        score = self.score_head(embed).squeeze(-1)
        # Mask padding moves with -inf so they vanish from the softmax pool.
        score = torch.where(
            candidates.mask > 0,
            score,
            score.new_full(score.shape, float("-inf")),
        )
        return embed, score


@dataclass(frozen=True)
class ForcednessSummary:
    pooled: torch.Tensor       # (B, embed_dim)
    entropy: torch.Tensor      # (B,)
    top1_mass: torch.Tensor    # (B,)
    gap: torch.Tensor          # (B,)
    check_mass: torch.Tensor   # (B,)
    promotion_mass: torch.Tensor      # (B,)
    underpromotion_mass: torch.Tensor # (B,)
    pin_mass: torch.Tensor            # (B,)
    capture_mass: torch.Tensor        # (B,)
    king_zone_mass: torch.Tensor      # (B,)
    overflow_count: torch.Tensor      # (B,)
    top_move_kind: torch.Tensor       # (B, MOVE_KIND_COUNT)
    candidate_count: torch.Tensor     # (B,)


class TopKMovePool(nn.Module):
    """Top-k softmax bottleneck over candidate scores."""

    def __init__(self, top_k: int, temperature: float = 1.0) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = int(top_k)
        self.log_temperature = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(temperature)))))

    def forward(
        self,
        embeddings: torch.Tensor,         # (B, K, embed_dim)
        scores: torch.Tensor,             # (B, K)
        candidates: CandidateMoves,
    ) -> ForcednessSummary:
        batch, k, embed_dim = embeddings.shape
        top_k = min(self.top_k, k)
        topk = torch.topk(scores, k=top_k, dim=-1)
        top_scores = topk.values         # (B, top_k)
        top_indices = topk.indices       # (B, top_k)
        gather_idx = top_indices.unsqueeze(-1).expand(batch, top_k, embed_dim)
        top_embed = torch.gather(embeddings, 1, gather_idx)
        top_flags = torch.gather(
            candidates.flags,
            1,
            top_indices.unsqueeze(-1).expand(batch, top_k, MOVE_FLAG_COUNT),
        )
        top_kinds = torch.gather(
            candidates.kind_onehot,
            1,
            top_indices.unsqueeze(-1).expand(batch, top_k, MOVE_KIND_COUNT),
        )
        valid_mask = torch.gather(candidates.mask, 1, top_indices)  # (B, top_k)

        # Convert -inf scores from padding into -inf so they vanish under softmax.
        valid_scores = torch.where(
            valid_mask > 0,
            top_scores,
            top_scores.new_full(top_scores.shape, float("-inf")),
        )
        temperature = self.log_temperature.exp().clamp(min=1e-2, max=1.0e2)
        weights = torch.softmax(valid_scores / temperature, dim=-1)
        # When all candidates are masked, softmax of all -inf is NaN; fall
        # back to uniform on the valid set, or zeros if no valid candidates.
        nan_mask = torch.isnan(weights).any(dim=-1, keepdim=True)
        weights = torch.where(
            nan_mask.expand_as(weights),
            valid_mask / valid_mask.sum(dim=-1, keepdim=True).clamp_min(1.0),
            weights,
        )
        weights = weights * valid_mask
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / weight_sum

        pooled = (weights.unsqueeze(-1) * top_embed).sum(dim=1)
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
        sorted_scores = torch.sort(valid_scores, dim=-1, descending=True).values
        finite = torch.isfinite(sorted_scores)
        first = torch.where(
            finite[:, 0],
            sorted_scores[:, 0],
            sorted_scores.new_zeros(batch),
        )
        if sorted_scores.shape[1] > 1:
            second = torch.where(
                finite[:, 1],
                sorted_scores[:, 1],
                first,
            )
        else:
            second = first
        gap = (first - second).clamp_min(0.0)
        top1_mass = weights.amax(dim=-1)

        # Flag masses across top_k under the pool weights.
        check_mass = (weights * top_flags[..., 0]).sum(dim=-1)
        capture_mass = (weights * top_flags[..., 1]).sum(dim=-1)
        # source_pinned and pin_aligned flags at indices 2, 3.
        pin_mass = (weights * (top_flags[..., 2] + top_flags[..., 3])).sum(dim=-1).clamp(0.0, 1.0)
        king_zone_mass = (weights * top_flags[..., 4]).sum(dim=-1)
        promotion_mass = (weights * top_flags[..., 7]).sum(dim=-1)
        underpromotion_mass = (weights * top_flags[..., 8]).sum(dim=-1)
        # The top-1 move's kind onehot under the pool weights (continuous proxy).
        top_move_kind = (weights.unsqueeze(-1) * top_kinds).sum(dim=1)

        overflow_count = (candidates.mask.sum(dim=-1) >= float(k)).to(scores.dtype)
        candidate_count = candidates.mask.sum(dim=-1)

        return ForcednessSummary(
            pooled=pooled,
            entropy=entropy,
            top1_mass=top1_mass,
            gap=gap,
            check_mass=check_mass,
            promotion_mass=promotion_mass,
            underpromotion_mass=underpromotion_mass,
            pin_mass=pin_mass,
            capture_mass=capture_mass,
            king_zone_mass=king_zone_mass,
            overflow_count=overflow_count,
            top_move_kind=top_move_kind,
            candidate_count=candidate_count,
        )


class CandidateMoveForcednessSheafNet(OrientedTacticalSheafNet):
    """i018 wrapped with a candidate-move forcedness bottleneck.

    The trunk forward is the i018 forward; the new move branch reads the
    final per-square states, the relation masks, and a small set of trunk
    diagnostics (`r`), and emits a gated additive logit delta plus a bundle
    of forcedness diagnostics. The delta head and gate output layers are
    zero-initialized, so the model is numerically equivalent to i018 at
    init (the only drift is FP32 reduction noise from the zero-times-x
    multiplications, which is well under `1e-6`).
    """

    def __init__(
        self,
        *args: Any,
        max_candidates: int = 96,
        top_k: int = 8,
        move_embed_dim: int = 48,
        move_hidden_dim: int = 64,
        delta_hidden_dim: int = 48,
        gate_hidden_dim: int = 24,
        softmax_temperature: float = 1.0,
        flat_move_pool: bool = False,
        disable_move_branch: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_candidates = int(max_candidates)
        self.top_k = int(top_k)
        self.flat_move_pool = bool(flat_move_pool)
        self.disable_move_branch = bool(disable_move_branch)
        d_model = self.blocks[0].node_to_stalk.in_features
        relation_count = len(RELATION_NAMES)
        self.move_builder = CandidateMoveBuilder(max_candidates=max_candidates)
        self.move_sheaf_summary = MoveLocalSheafSummary(relation_count=relation_count)
        self.move_encoder = CandidateMoveEncoder(
            d_model=d_model,
            sheaf_summary_dim=self.move_sheaf_summary.output_dim,
            embed_dim=int(move_embed_dim),
            hidden_dim=int(move_hidden_dim),
            dropout=float(kwargs.get("dropout", 0.1)),
        )
        self.move_pool = TopKMovePool(top_k=top_k, temperature=float(softmax_temperature))

        # Forcedness summary feature width passed to the delta + gate heads:
        #   pooled embedding (move_embed_dim)
        #   forcedness scalars: entropy, top1_mass, gap, check_mass,
        #     promotion_mass, underpromotion_mass, pin_mass, capture_mass,
        #     king_zone_mass, overflow_count, candidate_count -> 11 scalars
        #   trunk context scalars: sheaf_tension, ray_language_energy,
        #     triad_defect_energy, pin_pressure, king_ring_pressure,
        #     transport_imbalance, defense_gap, reply_pressure -> 8 scalars
        forced_scalar_count = 11
        trunk_scalar_count = 8
        delta_input_dim = int(move_embed_dim) + forced_scalar_count + trunk_scalar_count + MOVE_KIND_COUNT
        gate_input_dim = forced_scalar_count + trunk_scalar_count + MOVE_KIND_COUNT
        self.delta_head = nn.Sequential(
            nn.LayerNorm(delta_input_dim),
            nn.Linear(delta_input_dim, int(delta_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(delta_hidden_dim), 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), 1),
        )
        # Zero-init final layers so the model starts as i018 at init.
        for module in (self.delta_head[-1], self.gate_head[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def _trunk_scalar_bundle(
        self,
        diagnostics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        keys = (
            "sheaf_tension",
            "ray_language_energy",
            "triad_defect_energy",
            "pin_pressure",
            "king_ring_pressure",
            "transport_imbalance",
            "defense_gap",
            "reply_pressure",
        )
        return torch.stack([diagnostics[key] for key in keys], dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = require_board_tensor(x, self.spec)
        board = self.adapter(x)
        incidence = self.incidence(board.piece_state, board.occupancy)
        if self.scramble_relations:
            sheaf_masks = incidence.relation_masks
            batch_dim, relations_dim, squares_dim, _ = sheaf_masks.shape
            perm = torch.argsort(
                torch.rand(batch_dim, relations_dim, squares_dim, device=sheaf_masks.device),
                dim=-1,
            )
            perm_expanded = perm.unsqueeze(-2).expand(-1, -1, squares_dim, -1)
            scrambled_masks = torch.gather(sheaf_masks, dim=-1, index=perm_expanded)
        else:
            scrambled_masks = incidence.relation_masks

        h = self.encoder(board.square_raw, board.piece_state)
        block_energies: list[torch.Tensor] = []
        block_gates: list[torch.Tensor] = []
        for block in self.blocks:
            h, energy, gates = block(h, scrambled_masks)
            block_energies.append(energy)
            block_gates.append(gates.unsqueeze(0).expand(x.shape[0], -1))

        energy_stack = torch.stack(block_energies, dim=1)
        gate_stack = torch.stack(block_gates, dim=1)
        energy_mean = energy_stack.mean(dim=1)
        energy_max = energy_stack.amax(dim=1)
        gate_mean = gate_stack.mean(dim=1)
        triad_stats = (
            self.triad_pool(h, incidence)
            if self.triad_pool is not None
            else h.new_zeros(h.shape[0], 0)
        )
        readout = torch.cat(
            [
                h.mean(dim=1),
                h.amax(dim=1),
                _weighted_mean(h, incidence.our_piece),
                _weighted_mean(h, incidence.them_piece),
                energy_mean,
                energy_max,
                incidence.relation_density,
                gate_mean,
                triad_stats,
                self._board_stats(board, incidence),
            ],
            dim=1,
        )
        base_logits = _format_logits(self.head(readout), self.num_classes)
        sheaf_tension = energy_stack.mean(dim=(1, 2))
        us_pressure = incidence.relation_masks[:, 0].sum(dim=(1, 2))
        them_pressure = incidence.relation_masks[:, 1].sum(dim=(1, 2))
        us_defense = incidence.relation_masks[:, 2].sum(dim=(1, 2))
        them_defense = incidence.relation_masks[:, 3].sum(dim=(1, 2))
        rank_counts = torch.matmul(board.occupancy, self.incidence.rank_one_hot)
        file_counts = torch.matmul(board.occupancy, self.incidence.file_one_hot)
        piece_entropy = -(board.piece_state * board.piece_state.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=1)

        diagnostics: dict[str, torch.Tensor] = {
            "logits": base_logits,
            "mechanism_energy": torch.log1p(sheaf_tension),
            "proposal_profile_strength": gate_mean.mean(dim=1),
            "proposal_keyword_count": base_logits.new_full((x.shape[0],), 4.0),
            "sheaf_tension": sheaf_tension,
            "transport_imbalance": (us_pressure - them_pressure).abs() / (us_pressure + them_pressure).clamp_min(1.0),
            "symmetry_residual": (incidence.our_attack.mean(dim=(1, 2)) - incidence.them_attack.mean(dim=(1, 2))).abs(),
            "topology_pressure": incidence.relation_density.mean(dim=1),
            "ray_language_energy": energy_mean[:, 6:9].mean(dim=1),
            "information_surprisal": piece_entropy,
            "sparse_certificate_energy": energy_stack.amax(dim=(1, 2)),
            "rank_file_imbalance": (rank_counts.std(dim=1) - file_counts.std(dim=1)).abs(),
            "king_ring_pressure": incidence.relation_density[:, 4] + incidence.relation_density[:, 5],
            "reply_pressure": 0.5 * (us_pressure + them_pressure) / 64.0,
            "defense_gap": ((us_pressure + them_pressure) - (us_defense + them_defense)) / 64.0,
            "triad_defect_energy": triad_stats[:, 0] if triad_stats.numel() else base_logits.new_zeros(x.shape[0]),
            "pin_pressure": incidence.relation_density[:, 11],
        }

        if self.disable_move_branch:
            # i018-equivalent path: no candidate bottleneck, base logit only.
            diagnostics.update(self._zero_move_diagnostics(x.shape[0], base_logits.device, base_logits.dtype))
            return diagnostics

        candidates = self.move_builder(
            piece_state=board.piece_state,
            occupancy=board.occupancy,
            between=self.incidence.between,
            knight_geometry=self.incidence.knight,
            king_geometry=self.incidence.king,
            rook_geometry=self.incidence.rook_ray,
            bishop_geometry=self.incidence.bishop_ray,
            our_attack=incidence.our_attack,
            them_attack=incidence.them_attack,
            pin_mask=incidence.pin_mask,
        )
        sheaf_summary = self.move_sheaf_summary(incidence.relation_masks, candidates)
        embeddings, scores = self.move_encoder(h, candidates, sheaf_summary)
        if self.flat_move_pool:
            scores = torch.where(
                candidates.mask > 0,
                scores.new_zeros(scores.shape),
                scores.new_full(scores.shape, float("-inf")),
            )
        forced = self.move_pool(embeddings, scores, candidates)

        trunk_bundle = self._trunk_scalar_bundle(diagnostics)
        forced_scalars = torch.stack(
            [
                forced.entropy,
                forced.top1_mass,
                forced.gap,
                forced.check_mass,
                forced.promotion_mass,
                forced.underpromotion_mass,
                forced.pin_mass,
                forced.capture_mass,
                forced.king_zone_mass,
                forced.overflow_count,
                forced.candidate_count / max(1.0, float(self.max_candidates)),
            ],
            dim=-1,
        )
        gate_input = torch.cat([forced_scalars, trunk_bundle, forced.top_move_kind], dim=-1)
        delta_input = torch.cat([forced.pooled, forced_scalars, trunk_bundle, forced.top_move_kind], dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_input).squeeze(-1))
        delta = self.delta_head(delta_input).squeeze(-1)
        final_logits = base_logits + gate * delta

        diagnostics["logits"] = final_logits
        diagnostics["candidate_base_logits"] = base_logits
        diagnostics["candidate_delta_logits"] = delta
        diagnostics["candidate_gate"] = gate
        diagnostics["candidate_entropy"] = forced.entropy
        diagnostics["candidate_top1_mass"] = forced.top1_mass
        diagnostics["candidate_gap"] = forced.gap
        diagnostics["candidate_check_mass"] = forced.check_mass
        diagnostics["candidate_promotion_mass"] = forced.promotion_mass
        diagnostics["candidate_underpromotion_mass"] = forced.underpromotion_mass
        diagnostics["candidate_pin_mass"] = forced.pin_mass
        diagnostics["candidate_capture_mass"] = forced.capture_mass
        diagnostics["candidate_king_zone_mass"] = forced.king_zone_mass
        diagnostics["candidate_overflow_count"] = forced.overflow_count
        diagnostics["candidate_count"] = forced.candidate_count
        return diagnostics

    def _zero_move_diagnostics(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        zero = torch.zeros(batch, device=device, dtype=dtype)
        return {
            "candidate_base_logits": zero,
            "candidate_delta_logits": zero,
            "candidate_gate": zero,
            "candidate_entropy": zero,
            "candidate_top1_mass": zero,
            "candidate_gap": zero,
            "candidate_check_mass": zero,
            "candidate_promotion_mass": zero,
            "candidate_underpromotion_mass": zero,
            "candidate_pin_mass": zero,
            "candidate_capture_mass": zero,
            "candidate_king_zone_mass": zero,
            "candidate_overflow_count": zero,
            "candidate_count": zero,
        }


def build_candidate_move_forcedness_sheaf_from_config(
    config: dict[str, Any],
) -> CandidateMoveForcednessSheafNet:
    return CandidateMoveForcednessSheafNet(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 1)),
        channels=int(config.get("channels", 64)),
        hidden_dim=int(config.get("hidden_dim", 96)),
        depth=int(config.get("sheaf_layers", config.get("depth", 2))),
        stalk_dim=int(config.get("stalk_dim", 8)),
        dropout=float(config.get("dropout", 0.1)),
        encoding=str(config.get("encoding", "simple_18")),
        piece_adapter=str(config.get("piece_adapter", "exact")),
        use_triads=bool(config.get("use_triads", True)),
        scramble_relations=bool(config.get("scramble_relations", False)),
        max_candidates=int(config.get("max_candidates", 96)),
        top_k=int(config.get("top_k", 8)),
        move_embed_dim=int(config.get("move_embed_dim", 48)),
        move_hidden_dim=int(config.get("move_hidden_dim", 64)),
        delta_hidden_dim=int(config.get("delta_hidden_dim", 48)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 24)),
        softmax_temperature=float(config.get("softmax_temperature", 1.0)),
        flat_move_pool=bool(config.get("flat_move_pool", False)),
        disable_move_branch=bool(config.get("disable_move_branch", False)),
    )
