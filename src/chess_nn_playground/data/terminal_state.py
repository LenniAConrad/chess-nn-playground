"""Terminal-State Detection Primitive (TSDP) features.

The 11-dim rule-exact terminal feature vector defined in
``ideas/research/primitives/claude_05_terminal_state_detection.md`` and the
prototype ``ideas/research/primitives/prototypes/tsdp_prototype.py``.

All features are rule-derived from the current FEN/board state via
``python-chess``. No CRTK metadata, source labels, verification flags, or
engine scores are consulted.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import chess
import torch

from chess_nn_playground.data.board_features import (
    CASTLING_PLANES,
    EN_PASSANT_PLANE,
    NUM_INPUT_PLANES,
    PIECE_PLANES,
    SIDE_TO_MOVE_PLANE,
    SIMPLE_18,
)


TSDP_FEATURE_NAMES: tuple[str, ...] = (
    "mate_in_1",
    "mate_count",
    "stalemate_threat",
    "stalemate_count",
    "check_count",
    "promotion_count",
    "capture_count",
    "castling_count",
    "total_legal_moves",
    "forcing_density",
    "mating_special_count",
)
TSDP_FEATURE_DIM: int = len(TSDP_FEATURE_NAMES)


def _zero_features(dtype: np.dtype) -> np.ndarray:
    return np.zeros(TSDP_FEATURE_DIM, dtype=dtype)


def compute_terminal_state_features(
    board: str | chess.Board,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Compute the TSDP 11-dim feature vector for a single position.

    The position is identified either by a FEN string or a ``chess.Board``
    instance. The returned vector follows ``TSDP_FEATURE_NAMES`` order:

    - mate_in_1: 1.0 if any legal move delivers checkmate, else 0.0
    - mate_count: number of mating moves
    - stalemate_threat: 1.0 if any legal move stalemates opponent, else 0.0
    - stalemate_count: number of stalemate-producing moves
    - check_count: number of moves that give check but not mate
    - promotion_count: number of legal moves that are promotions
    - capture_count: number of legal moves that are captures
    - castling_count: number of legal castling moves
    - total_legal_moves: total legal move count
    - forcing_density: (check_count + capture_count) / max(total, 1)
    - mating_special_count: mating moves that are also promotion or capture

    Terminal positions (no legal moves) return zeros — the trunk already knows
    the side-to-move is mated or stalemated; TSDP only describes the moves the
    side-to-move can pick.
    """

    if isinstance(board, str):
        b = chess.Board(board)
    elif isinstance(board, chess.Board):
        b = board.copy(stack=False)
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported board input: {type(board)!r}")

    legal_moves = list(b.legal_moves)
    total = len(legal_moves)
    if total == 0:
        return _zero_features(dtype)

    mate_count = 0
    stalemate_count = 0
    check_count = 0
    promotion_count = 0
    capture_count = 0
    castling_count = 0
    mating_special_count = 0

    for move in legal_moves:
        is_promotion = move.promotion is not None
        is_capture = b.is_capture(move)
        is_castling = b.is_castling(move)

        b.push(move)
        is_checkmate = b.is_checkmate()
        is_stalemate = b.is_stalemate()
        is_check_only = b.is_check() and not is_checkmate
        b.pop()

        if is_checkmate:
            mate_count += 1
            if is_promotion or is_capture:
                mating_special_count += 1
        if is_stalemate:
            stalemate_count += 1
        if is_check_only:
            check_count += 1
        if is_promotion:
            promotion_count += 1
        if is_capture:
            capture_count += 1
        if is_castling:
            castling_count += 1

    forcing_density = (check_count + capture_count) / max(total, 1)
    features = np.array(
        [
            1.0 if mate_count > 0 else 0.0,
            float(mate_count),
            1.0 if stalemate_count > 0 else 0.0,
            float(stalemate_count),
            float(check_count),
            float(promotion_count),
            float(capture_count),
            float(castling_count),
            float(total),
            float(forcing_density),
            float(mating_special_count),
        ],
        dtype=dtype,
    )
    return features


def compute_terminal_state_features_batch(
    fens: Iterable[str],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Compute the (B, 11) feature matrix for a batch of FEN strings."""

    out = [compute_terminal_state_features(fen, dtype=dtype) for fen in fens]
    if not out:
        return np.zeros((0, TSDP_FEATURE_DIM), dtype=dtype)
    return np.stack(out, axis=0)


_PLANE_TO_PIECE: dict[int, tuple[chess.PieceType, chess.Color]] = {
    0: (chess.PAWN, chess.WHITE),
    1: (chess.KNIGHT, chess.WHITE),
    2: (chess.BISHOP, chess.WHITE),
    3: (chess.ROOK, chess.WHITE),
    4: (chess.QUEEN, chess.WHITE),
    5: (chess.KING, chess.WHITE),
    6: (chess.PAWN, chess.BLACK),
    7: (chess.KNIGHT, chess.BLACK),
    8: (chess.BISHOP, chess.BLACK),
    9: (chess.ROOK, chess.BLACK),
    10: (chess.QUEEN, chess.BLACK),
    11: (chess.KING, chess.BLACK),
}


def _plane_to_square(rank_idx: int, file_idx: int) -> int:
    # simple_18 stores piece plane at index ``[12 - rank, file]`` (rank 8 first).
    file = file_idx
    rank = 7 - rank_idx
    return chess.square(file, rank)


def simple_18_to_board(plane: np.ndarray | torch.Tensor) -> chess.Board:
    """Reconstruct a ``chess.Board`` from a single simple_18 tensor.

    The simple_18 encoding stores piece placement, side-to-move, all four
    castling rights, and the en-passant square — enough state to enumerate
    legal moves exactly. Halfmove and fullmove counters are not preserved by
    simple_18 and default to 0 / 1 here; tactical primitives like TSDP only
    care about legal moves and check/mate/stalemate detection, which are not
    affected by the missing counters in puzzle positions.
    """

    if isinstance(plane, torch.Tensor):
        array = plane.detach().cpu().numpy()
    else:
        array = np.asarray(plane)
    if array.shape != (NUM_INPUT_PLANES, 8, 8):
        raise ValueError(
            f"Expected simple_18 tensor of shape ({NUM_INPUT_PLANES}, 8, 8), got {tuple(array.shape)}"
        )

    board = chess.Board.empty()
    for plane_idx, (piece_type, color) in _PLANE_TO_PIECE.items():
        rows, cols = np.where(array[plane_idx] > 0.5)
        for rank_idx, file_idx in zip(rows.tolist(), cols.tolist()):
            board.set_piece_at(_plane_to_square(rank_idx, file_idx), chess.Piece(piece_type, color))

    side_to_move_plane = PIECE_PLANES.__len__()  # = 12
    board.turn = bool(array[side_to_move_plane].mean() > 0.5)

    castling_index = side_to_move_plane + 1  # 13
    castling_fen_parts: list[str] = []
    castling_rights_flags = [
        ("K", castling_index + 0),
        ("Q", castling_index + 1),
        ("k", castling_index + 2),
        ("q", castling_index + 3),
    ]
    for symbol, plane_position in castling_rights_flags:
        if float(array[plane_position].mean()) > 0.5:
            castling_fen_parts.append(symbol)
    castling_fen = "".join(castling_fen_parts) if castling_fen_parts else "-"
    try:
        board.set_castling_fen(castling_fen)
    except ValueError:
        # If the encoding produced rights that don't match piece placement,
        # drop them rather than crashing. TSDP cares about legal-move counts
        # which python-chess will recompute consistent with placement.
        board.set_castling_fen("-")

    en_passant_index = castling_index + len(CASTLING_PLANES)  # 17
    ep_plane = array[en_passant_index]
    ep_hits = np.where(ep_plane > 0.5)
    if ep_hits[0].size > 0:
        rank_idx = int(ep_hits[0][0])
        file_idx = int(ep_hits[1][0])
        board.ep_square = _plane_to_square(rank_idx, file_idx)
    else:
        board.ep_square = None

    return board


def simple_18_batch_to_terminal_state_features(
    batch: torch.Tensor,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Compute TSDP features for a batch of simple_18 tensors.

    Documented temporary fallback: each sample triggers ``python-chess`` legal
    move generation. The follow-up production path is a precomputed parquet
    column built by ``scripts/data/precompute_primitive_features.py`` (planned
    upgrade), which would short-circuit this CPU work. Until that lands, the
    model decodes the FEN state directly from the simple_18 tensor in
    ``forward``.
    """

    if batch.ndim != 4 or batch.shape[1] != NUM_INPUT_PLANES:
        raise ValueError(
            f"Expected simple_18 tensor with shape (B, {NUM_INPUT_PLANES}, 8, 8), got {tuple(batch.shape)}"
        )
    detached = batch.detach().cpu().numpy()
    feats = np.zeros((detached.shape[0], TSDP_FEATURE_DIM), dtype=dtype)
    for index in range(detached.shape[0]):
        board = simple_18_to_board(detached[index])
        feats[index] = compute_terminal_state_features(board, dtype=dtype)
    return feats


# Documented inputs assumed in the upstream encoding contract; not the model
# input names. See ``board_features.py`` for the canonical plane order.
__all__ = (
    "TSDP_FEATURE_NAMES",
    "TSDP_FEATURE_DIM",
    "compute_terminal_state_features",
    "compute_terminal_state_features_batch",
    "simple_18_batch_to_terminal_state_features",
    "simple_18_to_board",
)


_ = (SIDE_TO_MOVE_PLANE, EN_PASSANT_PLANE, SIMPLE_18)  # plane-name docs anchor
