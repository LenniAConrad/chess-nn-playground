"""Shared rule-derived graph features for the ray-cast / legal-move primitive batch.

Helpers in this module are deterministic functions of the simple_18 board
tensor. They never consult CRTK metadata, source labels, verification flags,
Stockfish scores, PVs, or report-only metadata.

The geometry tables (geometric attacks, between-square masks, ray steps) are
precomputed once at module-init time and reused across primitives p006-p011.
The same geometric tables also appear in the i193
``DualStreamFeatureBuilder``; we recompute them here so each primitive owns
its own tables and can be imported without pulling the full i193 module into
test imports.

Plane layout (matches ``chess_nn_playground.data.board_features``):

- 0..5  white piece planes in order (P, N, B, R, Q, K)
- 6..11 black piece planes in order (P, N, B, R, Q, K)
- 12    side-to-move (white_to_move)
- 13..16 castling rights
- 17    en-passant
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


WHITE = 0
BLACK = 1
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5
SQUARES = 64
NUM_PIECE_TYPES = 6
NUM_DIRECTIONS = 8

# Direction order is fixed: N, NE, E, SE, S, SW, W, NW. Used by ray
# primitives to index per-direction features.
DIR_NAMES: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DIR_VECTORS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
MAX_RAY_LEN = 7  # Maximum number of steps along any 8x8 ray.


def _square(row: int, file: int) -> int:
    return row * 8 + file


def _row_file(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def _inside(row: int, file: int) -> bool:
    return 0 <= row < 8 and 0 <= file < 8


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _piece_channel(color: int, piece: int) -> int:
    return piece if color == WHITE else 6 + piece


@dataclass(frozen=True)
class RuleGeometry:
    """Precomputed geometric tables shared across primitives.

    Tables are buffers (CPU tensors at module-init); each model should move
    them to its working device once and reuse the cached copy.
    """

    geom_attacks: torch.Tensor   # (6, 2, 64, 64) — piece type, color, source, target
    between: torch.Tensor        # (64, 64, 64) — squares between source and target
    ray_step_target: torch.Tensor  # (64, 8, 7) — long, target square at each ray step
    ray_step_valid: torch.Tensor   # (64, 8, 7) — bool, 1 where the step is on board
    ray_step_count: torch.Tensor   # (64, 8) — long, number of valid steps per ray


def _build_geometry() -> RuleGeometry:
    geom_attacks = torch.zeros(NUM_PIECE_TYPES, 2, SQUARES, SQUARES, dtype=torch.float32)
    between = torch.zeros(SQUARES, SQUARES, SQUARES, dtype=torch.float32)
    ray_step_target = torch.zeros(SQUARES, NUM_DIRECTIONS, MAX_RAY_LEN, dtype=torch.long)
    ray_step_valid = torch.zeros(SQUARES, NUM_DIRECTIONS, MAX_RAY_LEN, dtype=torch.bool)
    ray_step_count = torch.zeros(SQUARES, NUM_DIRECTIONS, dtype=torch.long)
    knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    king_offsets = [(r, f) for r in (-1, 0, 1) for f in (-1, 0, 1) if r != 0 or f != 0]
    bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for source in range(SQUARES):
        sr, sf = _row_file(source)
        for target in range(SQUARES):
            tr, tf = _row_file(target)
            if source == target:
                continue
            aligned = (sr == tr) or (sf == tf) or (abs(tr - sr) == abs(tf - sf))
            if aligned:
                row_step = _sign(tr - sr)
                file_step = _sign(tf - sf)
                row, file = sr + row_step, sf + file_step
                while (row, file) != (tr, tf):
                    between[source, target, _square(row, file)] = 1.0
                    row += row_step
                    file += file_step
        for color in (WHITE, BLACK):
            pawn_forward = -1 if color == WHITE else 1
            for fd in (-1, 1):
                r, f = sr + pawn_forward, sf + fd
                if _inside(r, f):
                    geom_attacks[PAWN, color, source, _square(r, f)] = 1.0
            for rd, fd in knight_offsets:
                r, f = sr + rd, sf + fd
                if _inside(r, f):
                    geom_attacks[KNIGHT, color, source, _square(r, f)] = 1.0
            for rd, fd in king_offsets:
                r, f = sr + rd, sf + fd
                if _inside(r, f):
                    geom_attacks[KING, color, source, _square(r, f)] = 1.0
            for piece, dirs in (
                (BISHOP, bishop_dirs),
                (ROOK, rook_dirs),
                (QUEEN, bishop_dirs + rook_dirs),
            ):
                for rd, fd in dirs:
                    r, f = sr + rd, sf + fd
                    while _inside(r, f):
                        geom_attacks[piece, color, source, _square(r, f)] = 1.0
                        r += rd
                        f += fd
        for direction, (drow, dfile) in enumerate(DIR_VECTORS):
            row = sr + drow
            file = sf + dfile
            step = 0
            while _inside(row, file) and step < MAX_RAY_LEN:
                ray_step_target[source, direction, step] = _square(row, file)
                ray_step_valid[source, direction, step] = True
                row += drow
                file += dfile
                step += 1
            ray_step_count[source, direction] = step
    return RuleGeometry(
        geom_attacks=geom_attacks,
        between=between,
        ray_step_target=ray_step_target,
        ray_step_valid=ray_step_valid,
        ray_step_count=ray_step_count,
    )


_RULE_GEOMETRY = _build_geometry()


def rule_geometry() -> RuleGeometry:
    """Return the module-global ``RuleGeometry`` singleton."""

    return _RULE_GEOMETRY


def occupancy_from_board(board: torch.Tensor) -> torch.Tensor:
    """Total occupancy mask ``(B, 64)`` from the simple_18 board tensor.

    Sums the 12 piece planes, flattens, and clamps to [0, 1].
    """
    if board.ndim != 4 or board.shape[1] < 12:
        raise ValueError(
            f"Expected simple_18 board (B, 18, 8, 8), got {tuple(board.shape)}"
        )
    piece_planes = board[:, :12].flatten(2).clamp(0.0, 1.0)
    return piece_planes.sum(dim=1).clamp(0.0, 1.0)


def side_to_move_from_board(board: torch.Tensor) -> torch.Tensor:
    """Side-to-move flag ``(B,)`` from plane 12 (1 = white-to-move)."""

    return board[:, 12].mean(dim=(1, 2)).clamp(0.0, 1.0)


def piece_planes_flat(board: torch.Tensor) -> torch.Tensor:
    """Flattened piece planes ``(B, 12, 64)``."""

    return board[:, :12].flatten(2).clamp(0.0, 1.0)


def compute_attack_relations(
    board: torch.Tensor,
    geometry: RuleGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-color square-to-square attack relations with occlusion.

    Returns:
        attacks[color]: ``(B, 64, 64)`` — square ``i`` attacks square ``j``
            using a piece of side ``color`` on square ``i``. Sliding-piece
            relations are gated by the between-square occupancy.
        rays[color]: ``(B, 64, 64)`` — same as ``attacks`` but only for
            sliding pieces (B, R, Q). Useful for ray-aware primitives.
    """
    if board.ndim != 4 or board.shape[1] < 13:
        raise ValueError(
            f"Expected simple_18 board (B, 18, 8, 8), got {tuple(board.shape)}"
        )
    device = board.device
    dtype = board.dtype
    batch = board.shape[0]
    piece_planes = piece_planes_flat(board)
    occ = occupancy_from_board(board)

    between = geometry.between.to(device=device, dtype=dtype)
    geom = geometry.geom_attacks.to(device=device, dtype=dtype)

    blocked = torch.einsum("stk,bk->bst", between, occ)
    clear_slide = (blocked <= 0.5).to(dtype=dtype)
    clear_dense = torch.ones_like(clear_slide)

    attacks_per_color: list[torch.Tensor] = []
    rays_per_color: list[torch.Tensor] = []
    for color in (WHITE, BLACK):
        attack_sum = piece_planes.new_zeros(batch, SQUARES, SQUARES)
        ray_sum = piece_planes.new_zeros(batch, SQUARES, SQUARES)
        for piece in range(NUM_PIECE_TYPES):
            source = piece_planes[:, _piece_channel(color, piece)]
            line_clear = clear_slide if piece in {BISHOP, ROOK, QUEEN} else clear_dense
            relation = source[:, :, None] * geom[piece, color].unsqueeze(0) * line_clear
            attack_sum = attack_sum + relation
            if piece in {BISHOP, ROOK, QUEEN}:
                ray_sum = ray_sum + relation
        attacks_per_color.append(attack_sum)
        rays_per_color.append(ray_sum)
    return torch.stack(attacks_per_color, dim=1), torch.stack(rays_per_color, dim=1)


def select_by_side_to_move(
    white_tensor: torch.Tensor,
    black_tensor: torch.Tensor,
    stm: torch.Tensor,
) -> torch.Tensor:
    """Select white or black tensor per-sample by the side-to-move scalar."""
    selector = stm.view(-1, *([1] * (white_tensor.ndim - 1)))
    return selector * white_tensor + (1.0 - selector) * black_tensor


def compute_legal_move_graph(
    board: torch.Tensor,
    geometry: RuleGeometry,
) -> torch.Tensor:
    """Approximation of the legal-move graph as ``(B, 64, 64)``.

    Returns 1 on (source, target) pairs where the side-to-move has a
    pseudo-legal move from ``source`` to ``target``. The approximation
    drops a few subtleties (in-check counters, en-passant, castling) but
    captures the dominant chess connectivity: own piece can move (with
    occlusion for sliders) to ``target``, and ``target`` is not occupied by
    an own piece. This matches the "rule-derived sparse graph" contract
    that primitives in this batch consume.
    """
    attacks, _ = compute_attack_relations(board, geometry)
    stm = side_to_move_from_board(board)
    piece_planes = piece_planes_flat(board)
    own_pieces_white = piece_planes[:, :6].sum(dim=1).clamp(0.0, 1.0)
    own_pieces_black = piece_planes[:, 6:12].sum(dim=1).clamp(0.0, 1.0)
    own_pieces = select_by_side_to_move(own_pieces_white, own_pieces_black, stm)
    attack_own = select_by_side_to_move(attacks[:, WHITE], attacks[:, BLACK], stm)
    target_open = (1.0 - own_pieces).unsqueeze(1)  # (B, 1, 64)
    return (attack_own * target_open).clamp(0.0, 1.0)


def compute_ray_transmittance(
    board: torch.Tensor,
    geometry: RuleGeometry,
) -> torch.Tensor:
    """Per-square per-direction per-step transmittance ``(B, 64, 8, 7)``.

    Entry ``T[b, s, d, k]`` is the probability that a ray launched from
    square ``s`` in direction ``d`` reaches step ``k`` without hitting any
    occupied square. Computed in log-domain prefix sums for numerical
    stability; clamped occupancy keeps log finite. Step 0 is the very next
    square along the ray and always has transmittance 1.
    """
    if board.ndim != 4 or board.shape[1] < 12:
        raise ValueError(
            f"Expected simple_18 board (B, 18, 8, 8), got {tuple(board.shape)}"
        )
    device = board.device
    dtype = board.dtype
    occ = occupancy_from_board(board)  # (B, 64)

    ray_target = geometry.ray_step_target.to(device=device)
    ray_valid = geometry.ray_step_valid.to(device=device, dtype=dtype)
    eps = 1.0e-6

    flat_target = ray_target.view(-1)
    occ_along = occ.index_select(1, flat_target).view(occ.shape[0], SQUARES, NUM_DIRECTIONS, MAX_RAY_LEN)
    occ_along = occ_along * ray_valid

    # Probability not blocked at each step (1 - O), masked to 1 off-board.
    not_blocked = (1.0 - occ_along).clamp(eps, 1.0)
    log_not_blocked = not_blocked.log()
    log_prefix = log_not_blocked.cumsum(dim=-1)
    # Transmittance at step k is product over u<k of (1 - O[u]); shift by one.
    shifted = torch.cat(
        [log_prefix.new_zeros(*log_prefix.shape[:-1], 1), log_prefix[..., :-1]],
        dim=-1,
    )
    transmittance = shifted.exp() * ray_valid
    return transmittance


def first_blocker_indices(
    board: torch.Tensor,
    geometry: RuleGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each square / ray-direction, return the first blocker target square.

    Returns:
        target: ``(B, 64, 8)`` long — index of the first occupied square
            along the ray. If no piece is on the ray, the index points at
            the source square (a safe self-loop that downstream code can
            mask using ``has_blocker``).
        has_blocker: ``(B, 64, 8)`` bool — True if a blocker exists.
    """
    device = board.device
    occ = occupancy_from_board(board)
    ray_target = geometry.ray_step_target.to(device=device)
    ray_valid = geometry.ray_step_valid.to(device=device)
    ray_count = geometry.ray_step_count.to(device=device)

    flat_target = ray_target.view(-1)
    occ_along = occ.index_select(1, flat_target).view(occ.shape[0], SQUARES, NUM_DIRECTIONS, MAX_RAY_LEN)
    occ_bool = (occ_along > 0.5) & ray_valid

    # Find first True along the step axis using argmax on a forward-cumsum mask.
    step_idx = torch.arange(MAX_RAY_LEN, device=device).view(1, 1, 1, -1)
    masked_step = torch.where(
        occ_bool,
        step_idx.expand_as(occ_bool),
        torch.full_like(occ_bool, MAX_RAY_LEN, dtype=torch.long),
    )
    first_step, _ = masked_step.min(dim=-1)  # (B, 64, 8)
    has_blocker = first_step < ray_count.unsqueeze(0).expand_as(first_step)

    # Pick the target square at the first-step index; clamp invalid to 0.
    clamped_step = first_step.clamp(0, MAX_RAY_LEN - 1)
    target = ray_target.unsqueeze(0).expand(occ.shape[0], -1, -1, -1).gather(
        -1, clamped_step.unsqueeze(-1)
    ).squeeze(-1)
    self_idx = torch.arange(SQUARES, device=device).view(1, SQUARES, 1).expand_as(target)
    target = torch.where(has_blocker, target, self_idx)
    return target, has_blocker


class SquareTokenEmbedder(nn.Module):
    """Per-square token embedder over the simple_18 board.

    Produces ``(B, 64, embed_dim)`` tokens by a per-square 1x1 conv followed
    by an optional GELU/LayerNorm. Cheap drop-in feature tower used by
    routing primitives that need square-level embeddings without paying for
    a full conv stack.
    """

    def __init__(
        self,
        input_channels: int = 18,
        embed_dim: int = 32,
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_channels) != 18:
            raise ValueError(
                "SquareTokenEmbedder requires the simple_18 board tensor"
            )
        self.input_channels = int(input_channels)
        self.embed_dim = int(embed_dim)
        hidden = int(hidden_dim) if int(hidden_dim) > 0 else int(embed_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(int(input_channels), hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, int(embed_dim), kernel_size=1, bias=True),
        )
        self.norm = nn.LayerNorm(int(embed_dim))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        if board.ndim != 4 or board.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected simple_18 board (B, {self.input_channels}, 8, 8), got {tuple(board.shape)}"
            )
        feat = self.proj(board)  # (B, embed_dim, 8, 8)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, 64, embed_dim)
        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)
        return tokens


__all__ = (
    "WHITE",
    "BLACK",
    "PAWN",
    "KNIGHT",
    "BISHOP",
    "ROOK",
    "QUEEN",
    "KING",
    "SQUARES",
    "NUM_PIECE_TYPES",
    "NUM_DIRECTIONS",
    "DIR_NAMES",
    "DIR_VECTORS",
    "MAX_RAY_LEN",
    "RuleGeometry",
    "rule_geometry",
    "occupancy_from_board",
    "side_to_move_from_board",
    "piece_planes_flat",
    "compute_attack_relations",
    "compute_legal_move_graph",
    "compute_ray_transmittance",
    "first_blocker_indices",
    "select_by_side_to_move",
    "SquareTokenEmbedder",
)
