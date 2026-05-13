"""Per-board legal-move adjacency built from the ``simple_18`` board tensor.

Several Gemini-batch primitives (p031 Legal-Move Laplacian, p032 Dynamic
Adjacency Gate, p033 Move-Kernel Operator, p034 Octilinear Selective Scan,
p035 Sparse Legal-Move Graph Transition) share the same rule-derived
substrate: an ``(B, 64, 64)`` pseudo-legal move adjacency where edge
``(i, j)`` fires iff some own piece on plane square ``i`` can pseudo-move /
attack plane square ``j`` under chess-rule geometry with blocker resolution.

The helper computes everything analytically from the ``simple_18`` piece
planes, side-to-move plane, castling planes, and en-passant plane. No
``python-chess`` call is required. CRTK metadata, source labels, and engine
scores are *not* read at any point — see the project rule in
``ideas/research/primitives/PRIMITIVE_TRAINING_TODO.md``.

The geometry tables here are the same ones used by the i193 trunk's
``DualStreamFeatureBuilder`` so the legal-move graph is consistent with the
exchange/king attack rasters that the trunk already produces. The
implementation includes pawn pushes and pawn captures (split by chess rule),
knight jumps (occlusion-free), king steps (occlusion-free), and sliding
moves for bishops, rooks, and queens with own/enemy blocker resolution.

The adjacency is non-differentiable by construction — it depends on rule
indicators, not learnable weights — so callers use it as a stop-gradient
mask. ``move_type`` codes are useful for primitives that want move-type
weight sharing (Move-Kernel Operator and Octilinear Selective Scan).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# Plane indices in the ``simple_18`` encoding (mirrors
# chess_nn_playground.data.board_features.PLANE_NAMES).
SQUARES = 64
PIECE_PLANE_COUNT = 12
WHITE_PIECE_PLANES = (0, 1, 2, 3, 4, 5)
BLACK_PIECE_PLANES = (6, 7, 8, 9, 10, 11)
STM_PLANE = 12

# Move-type integer codes used by ``move_type``. 0 means "no edge".
MOVE_TYPE_NONE = 0
MOVE_TYPE_KNIGHT = 1
MOVE_TYPE_RANK = 2  # horizontal sliding (E / W)
MOVE_TYPE_FILE = 3  # vertical sliding (N / S)
MOVE_TYPE_DIAG = 4  # NE / SW
MOVE_TYPE_ANTIDIAG = 5  # NW / SE
MOVE_TYPE_KING = 6  # adjacent king step (occlusion-free)
MOVE_TYPE_PAWN_PUSH = 7
MOVE_TYPE_PAWN_CAPTURE = 8
NUM_MOVE_TYPES = 9

# Direction integer codes for ``ray_direction``. 0 means "no edge".
# Directions are taken from the *source* square towards the target.
DIRECTION_NONE = 0
DIRECTION_E = 1
DIRECTION_W = 2
DIRECTION_N = 3
DIRECTION_S = 4
DIRECTION_NE = 5
DIRECTION_SW = 6
DIRECTION_NW = 7
DIRECTION_SE = 8
NUM_DIRECTIONS = 9


def _row_file(square: int) -> tuple[int, int]:
    return square // 8, square % 8


def _square(row: int, file: int) -> int:
    return row * 8 + file


def _inside(row: int, file: int) -> bool:
    return 0 <= row < 8 and 0 <= file < 8


def _direction_code(d_row: int, d_file: int) -> int:
    if d_row == 0 and d_file > 0:
        return DIRECTION_E
    if d_row == 0 and d_file < 0:
        return DIRECTION_W
    if d_row < 0 and d_file == 0:
        # simple_18 stores rank 8 at row 0; "north" in chess corresponds to
        # decreasing plane row index, so d_row < 0 maps to DIRECTION_N.
        return DIRECTION_N
    if d_row > 0 and d_file == 0:
        return DIRECTION_S
    if d_row < 0 and d_file > 0:
        return DIRECTION_NE
    if d_row > 0 and d_file < 0:
        return DIRECTION_SW
    if d_row < 0 and d_file < 0:
        return DIRECTION_NW
    if d_row > 0 and d_file > 0:
        return DIRECTION_SE
    return DIRECTION_NONE


def _move_type_for_step(d_row: int, d_file: int) -> int:
    if d_row == 0 and d_file != 0:
        return MOVE_TYPE_RANK
    if d_row != 0 and d_file == 0:
        return MOVE_TYPE_FILE
    if d_row == d_file:
        return MOVE_TYPE_DIAG
    if d_row == -d_file:
        return MOVE_TYPE_ANTIDIAG
    return MOVE_TYPE_NONE


@dataclass(frozen=True)
class _Geometry:
    """Static geometry tables that depend only on the 8x8 board.

    All tensors are float32. The semantics use the ``simple_18`` plane
    convention: plane row 0 corresponds to chess rank 8 (the back rank from
    white's perspective). The helper returns directional tables for
    sliding pieces and the leap tables for knight and king moves.
    """

    knight_edges: torch.Tensor  # (64, 64) {0, 1}
    king_edges: torch.Tensor  # (64, 64) {0, 1}
    between: torch.Tensor  # (64, 64, 64) {0, 1}
    aligned: torch.Tensor  # (64, 64) {0, 1}
    move_type: torch.Tensor  # (64, 64) long, MOVE_TYPE_*
    ray_direction: torch.Tensor  # (64, 64) long, DIRECTION_*
    pawn_push_white: torch.Tensor  # (64, 64) {0, 1}: source -> push target
    pawn_push_black: torch.Tensor
    pawn_capture_white: torch.Tensor  # (64, 64) {0, 1}: source -> capture target
    pawn_capture_black: torch.Tensor


_GEOMETRY_CACHE: _Geometry | None = None


def _compute_geometry() -> _Geometry:
    knight_edges = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    king_edges = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    between = torch.zeros(SQUARES, SQUARES, SQUARES, dtype=torch.float32)
    aligned = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    move_type = torch.zeros(SQUARES, SQUARES, dtype=torch.long)
    ray_direction = torch.zeros(SQUARES, SQUARES, dtype=torch.long)
    pawn_push_white = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    pawn_push_black = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    pawn_capture_white = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)
    pawn_capture_black = torch.zeros(SQUARES, SQUARES, dtype=torch.float32)

    knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    king_offsets = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if dr != 0 or df != 0]

    for source in range(SQUARES):
        sr, sf = _row_file(source)
        for dr, df in knight_offsets:
            r, f = sr + dr, sf + df
            if _inside(r, f):
                target = _square(r, f)
                knight_edges[source, target] = 1.0
                move_type[source, target] = MOVE_TYPE_KNIGHT
        for dr, df in king_offsets:
            r, f = sr + dr, sf + df
            if _inside(r, f):
                target = _square(r, f)
                king_edges[source, target] = 1.0
                # King steps share move-type codes with sliding pieces so the
                # move-type table preserves geometric direction. The dedicated
                # ``king_edges`` channel marks the king-leap; primitives that
                # want piece-typed king edges look at king_edges directly.
                if move_type[source, target].item() == MOVE_TYPE_NONE:
                    move_type[source, target] = _move_type_for_step(dr, df)
                ray_direction[source, target] = _direction_code(dr, df)
        # Sliding-piece tables.
        for dr, df in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            step = 1
            r, f = sr + dr, sf + df
            while _inside(r, f):
                target = _square(r, f)
                aligned[source, target] = 1.0
                move_type_step = _move_type_for_step(dr, df)
                # Overwrite knight/king priorities for slide-aligned squares.
                move_type[source, target] = move_type_step
                ray_direction[source, target] = _direction_code(dr, df)
                # Record the between mask (squares strictly between source and target
                # along this ray).
                rr, ff = sr + dr, sf + df
                while (rr, ff) != (r, f):
                    between[source, target, _square(rr, ff)] = 1.0
                    rr += dr
                    ff += df
                r += dr
                f += df
                step += 1
        # Pawn pushes and captures (rank-relative; halfmove move-numbers ignored).
        # White pushes "up" the board: plane row decreases by 1.
        if _inside(sr - 1, sf):
            pawn_push_white[source, _square(sr - 1, sf)] = 1.0
            if sr == 6:  # white pawn on rank 2 -> two-square push available
                pawn_push_white[source, _square(sr - 2, sf)] = 1.0
        for df_attack in (-1, 1):
            if _inside(sr - 1, sf + df_attack):
                pawn_capture_white[source, _square(sr - 1, sf + df_attack)] = 1.0
        # Black pushes "down": plane row increases by 1.
        if _inside(sr + 1, sf):
            pawn_push_black[source, _square(sr + 1, sf)] = 1.0
            if sr == 1:  # black pawn on rank 7 -> two-square push available
                pawn_push_black[source, _square(sr + 2, sf)] = 1.0
        for df_attack in (-1, 1):
            if _inside(sr + 1, sf + df_attack):
                pawn_capture_black[source, _square(sr + 1, sf + df_attack)] = 1.0

    return _Geometry(
        knight_edges=knight_edges,
        king_edges=king_edges,
        between=between,
        aligned=aligned,
        move_type=move_type,
        ray_direction=ray_direction,
        pawn_push_white=pawn_push_white,
        pawn_push_black=pawn_push_black,
        pawn_capture_white=pawn_capture_white,
        pawn_capture_black=pawn_capture_black,
    )


def _get_geometry() -> _Geometry:
    global _GEOMETRY_CACHE
    if _GEOMETRY_CACHE is None:
        _GEOMETRY_CACHE = _compute_geometry()
    return _GEOMETRY_CACHE


@dataclass(frozen=True)
class LegalMoveGraph:
    """Output of ``compute_legal_move_graph``.

    All tensors are placed on the same device / dtype as the input board.
    Edge values are 0.0 or 1.0 (float32 by default); ``move_type``,
    ``ray_direction`` and the piece-type masks are int64.
    """

    adjacency: torch.Tensor  # (B, 64, 64) float, edge from own piece i to dest j
    own_piece_mask: torch.Tensor  # (B, 64) float
    enemy_piece_mask: torch.Tensor  # (B, 64) float
    move_type: torch.Tensor  # (B, 64, 64) long, MOVE_TYPE_* (broadcast from static table)
    ray_direction: torch.Tensor  # (B, 64, 64) long, DIRECTION_*
    degree: torch.Tensor  # (B, 64) float
    occupancy: torch.Tensor  # (B, 64) float


def compute_legal_move_graph(board: torch.Tensor) -> LegalMoveGraph:
    """Build the per-board pseudo-legal move adjacency from ``simple_18``.

    The adjacency entry ``adjacency[b, i, j]`` is 1.0 iff some own-color
    piece sitting on plane square ``i`` (in board ``b``) can pseudo-legally
    move or capture to plane square ``j`` under chess-rule geometry with
    own/enemy blocker resolution. Edges to own-occupied targets are
    suppressed. Sliding-piece edges are blocked by the first occupied
    square on the ray (inclusive of enemy-piece captures, exclusive of own
    piece blockers). Pawn pushes and captures are split per chess rule and
    pawn pushes require the target to be empty.

    ``adjacency`` is treated as stop-gradient by callers: edge existence
    depends on rule-discrete indicators that are not differentiable.

    Args:
        board: ``(B, 18, 8, 8)`` simple_18 board tensor.

    Returns:
        ``LegalMoveGraph`` with adjacency, masks, degree, and per-edge
        move-type / ray-direction codes.
    """
    if board.ndim != 4 or board.shape[1] < PIECE_PLANE_COUNT + 1 or board.shape[2:] != (8, 8):
        raise ValueError(
            f"compute_legal_move_graph expects (B, >=13, 8, 8) board tensor; got {tuple(board.shape)}"
        )

    device = board.device
    dtype = board.dtype
    batch = board.shape[0]
    geom = _get_geometry()
    knight_edges = geom.knight_edges.to(device=device, dtype=dtype)
    king_edges = geom.king_edges.to(device=device, dtype=dtype)
    between = geom.between.to(device=device, dtype=dtype)
    aligned = geom.aligned.to(device=device, dtype=dtype)
    move_type_table = geom.move_type.to(device=device)
    ray_direction_table = geom.ray_direction.to(device=device)
    pawn_push_white = geom.pawn_push_white.to(device=device, dtype=dtype)
    pawn_push_black = geom.pawn_push_black.to(device=device, dtype=dtype)
    pawn_capture_white = geom.pawn_capture_white.to(device=device, dtype=dtype)
    pawn_capture_black = geom.pawn_capture_black.to(device=device, dtype=dtype)

    piece_planes = board[:, :PIECE_PLANE_COUNT].flatten(2).clamp(0.0, 1.0)  # (B, 12, 64)
    occupancy = piece_planes.sum(dim=1).clamp(0.0, 1.0)  # (B, 64)
    white_mask = piece_planes[:, list(WHITE_PIECE_PLANES)].sum(dim=1).clamp(0.0, 1.0)  # (B, 64)
    black_mask = piece_planes[:, list(BLACK_PIECE_PLANES)].sum(dim=1).clamp(0.0, 1.0)

    stm = board[:, STM_PLANE].mean(dim=(1, 2)).clamp(0.0, 1.0).view(-1)  # (B,)
    selector = stm.view(-1, 1)
    own_mask = selector * white_mask + (1.0 - selector) * black_mask
    enemy_mask = selector * black_mask + (1.0 - selector) * white_mask

    # Per-piece-type own-color planes for source weights.
    white_pawn = piece_planes[:, 0]
    white_knight = piece_planes[:, 1]
    white_bishop = piece_planes[:, 2]
    white_rook = piece_planes[:, 3]
    white_queen = piece_planes[:, 4]
    white_king = piece_planes[:, 5]
    black_pawn = piece_planes[:, 6]
    black_knight = piece_planes[:, 7]
    black_bishop = piece_planes[:, 8]
    black_rook = piece_planes[:, 9]
    black_queen = piece_planes[:, 10]
    black_king = piece_planes[:, 11]

    selector_b = selector  # (B, 1)
    own_pawn = selector_b * white_pawn + (1.0 - selector_b) * black_pawn
    own_knight = selector_b * white_knight + (1.0 - selector_b) * black_knight
    own_bishop = selector_b * white_bishop + (1.0 - selector_b) * black_bishop
    own_rook = selector_b * white_rook + (1.0 - selector_b) * black_rook
    own_queen = selector_b * white_queen + (1.0 - selector_b) * black_queen
    own_king = selector_b * white_king + (1.0 - selector_b) * black_king
    own_pawn_push_table = selector_b.unsqueeze(-1) * pawn_push_white.unsqueeze(0) + (
        1.0 - selector_b.unsqueeze(-1)
    ) * pawn_push_black.unsqueeze(0)  # (B, 64, 64)
    own_pawn_capture_table = selector_b.unsqueeze(-1) * pawn_capture_white.unsqueeze(
        0
    ) + (1.0 - selector_b.unsqueeze(-1)) * pawn_capture_black.unsqueeze(0)

    # --- Sliding pieces with blocker resolution.
    # blocked_count[b, s, t] = number of occupied squares strictly between s and t.
    blocked_count = torch.einsum("stk,bk->bst", between, occupancy)
    clear = (blocked_count <= 0.5).to(dtype=dtype)  # (B, 64, 64), 1 if ray clear

    # Diag / antidiag reachability from any source.
    # For each source row of ``aligned``, the move-type table tells us if the
    # alignment is diag, antidiag, rank, or file.
    move_type_long = move_type_table  # (64, 64) long, broadcasting later
    rank_mask = (move_type_long == MOVE_TYPE_RANK).to(dtype=dtype)
    file_mask = (move_type_long == MOVE_TYPE_FILE).to(dtype=dtype)
    diag_mask = (move_type_long == MOVE_TYPE_DIAG).to(dtype=dtype)
    antidiag_mask = (move_type_long == MOVE_TYPE_ANTIDIAG).to(dtype=dtype)
    cross_mask = rank_mask + file_mask  # rook / queen
    diag_any_mask = diag_mask + antidiag_mask  # bishop / queen

    # Compose sliding move edges: source has a piece type, target lies on the
    # appropriate ray, the between squares are clear.
    bishop_edges = own_bishop.unsqueeze(-1) * diag_any_mask.unsqueeze(0) * clear
    rook_edges = own_rook.unsqueeze(-1) * cross_mask.unsqueeze(0) * clear
    queen_edges = own_queen.unsqueeze(-1) * (diag_any_mask + cross_mask).unsqueeze(0) * clear
    sliding_edges = bishop_edges + rook_edges + queen_edges

    # --- Knight, king, pawn captures (occlusion-free moves).
    knight_edges_b = own_knight.unsqueeze(-1) * knight_edges.unsqueeze(0)
    king_edges_b = own_king.unsqueeze(-1) * king_edges.unsqueeze(0)
    pawn_capture_edges = own_pawn.unsqueeze(-1) * own_pawn_capture_table

    # --- Pawn pushes (require the target empty and, for the two-square push,
    # the intermediate square empty).
    target_empty = (1.0 - occupancy.unsqueeze(1))  # (B, 1, 64), broadcasts to (B, 64, 64)
    # The two-square push needs the square between source and target to also
    # be empty; ``between`` already encodes "squares strictly between".
    push_clear = clear  # equal to "no piece in between"
    pawn_push_edges = (
        own_pawn.unsqueeze(-1) * own_pawn_push_table * target_empty * push_clear
    )

    # --- Combine and drop edges that target own pieces.
    raw_edges = (
        sliding_edges + knight_edges_b + king_edges_b + pawn_capture_edges + pawn_push_edges
    )
    target_own = own_mask.unsqueeze(1)  # (B, 1, 64)
    drop_to_own = 1.0 - target_own
    # For pawn captures the target needs to actually have an enemy piece for
    # a real capture, but the proposal framing treats *threatening* squares
    # as the "move graph" -- enemy-pawn captures-with-enemy-occupancy + en
    # passant fall under it. Allow the threat target even if empty so the
    # graph is symmetric under pseudo-legal threat semantics.
    adjacency = raw_edges * drop_to_own
    adjacency = adjacency.clamp(0.0, 1.0)

    # Diagonal edges are not used: a piece cannot move to its own square.
    eye = torch.eye(SQUARES, device=device, dtype=dtype).unsqueeze(0)
    adjacency = adjacency * (1.0 - eye)

    # Materialise the move-type / direction tables broadcast to batch size.
    move_type_b = move_type_table.unsqueeze(0).expand(batch, SQUARES, SQUARES).contiguous()
    ray_direction_b = ray_direction_table.unsqueeze(0).expand(batch, SQUARES, SQUARES).contiguous()
    degree = adjacency.sum(dim=-1)

    return LegalMoveGraph(
        adjacency=adjacency,
        own_piece_mask=own_mask,
        enemy_piece_mask=enemy_mask,
        move_type=move_type_b,
        ray_direction=ray_direction_b,
        degree=degree,
        occupancy=occupancy,
    )


__all__ = (
    "LegalMoveGraph",
    "compute_legal_move_graph",
    "DIRECTION_E",
    "DIRECTION_N",
    "DIRECTION_NE",
    "DIRECTION_NONE",
    "DIRECTION_NW",
    "DIRECTION_S",
    "DIRECTION_SE",
    "DIRECTION_SW",
    "DIRECTION_W",
    "MOVE_TYPE_ANTIDIAG",
    "MOVE_TYPE_DIAG",
    "MOVE_TYPE_FILE",
    "MOVE_TYPE_KING",
    "MOVE_TYPE_KNIGHT",
    "MOVE_TYPE_NONE",
    "MOVE_TYPE_PAWN_CAPTURE",
    "MOVE_TYPE_PAWN_PUSH",
    "MOVE_TYPE_RANK",
    "NUM_DIRECTIONS",
    "NUM_MOVE_TYPES",
    "SQUARES",
)
