from __future__ import annotations

import numpy as np

from chess_nn_playground.data.fen_utils import parse_fen


PIECE_PLANES = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
SIDE_TO_MOVE_PLANE = "white_to_move"
CASTLING_PLANES = ["white_kingside", "white_queenside", "black_kingside", "black_queenside"]
EN_PASSANT_PLANE = "en_passant"
SIMPLE_18 = "simple_18"
LC0_STATIC_112 = "lc0_static_112"
LC0_BT4_112 = "lc0_bt4_112"

PLANE_NAMES = PIECE_PLANES + [SIDE_TO_MOVE_PLANE] + CASTLING_PLANES + [EN_PASSANT_PLANE]
NUM_INPUT_PLANES = len(PLANE_NAMES)
LC0_HISTORY_STEPS = 8
LC0_PLANES_PER_STEP = 13
LC0_STEP_PLANES = [
    "our_pawns",
    "our_knights",
    "our_bishops",
    "our_rooks",
    "our_queens",
    "our_king",
    "their_pawns",
    "their_knights",
    "their_bishops",
    "their_rooks",
    "their_queens",
    "their_king",
    "repetition",
]
LC0_AUX_PLANES = [
    "white_to_move",
    "black_to_move",
    "white_kingside",
    "white_queenside",
    "black_kingside",
    "black_queenside",
    "en_passant_square",
    "move_number_scaled",
]
LC0_STATIC_PLANE_NAMES = [
    f"history_{history}_{name}"
    for history in range(LC0_HISTORY_STEPS)
    for name in PIECE_PLANES + ["repetition"]
] + LC0_AUX_PLANES
LC0_BT4_AUX_PLANES = [
    "queenside_castling_rooks",
    "kingside_castling_rooks",
    "reserved_zero_0",
    "reserved_zero_1",
    "en_passant_file",
    "rule50_ply_scaled",
    "reserved_zero_2",
    "all_ones",
]
LC0_BT4_PLANE_NAMES = [
    f"history_{history}_{name}"
    for history in range(LC0_HISTORY_STEPS)
    for name in LC0_STEP_PLANES
] + LC0_BT4_AUX_PLANES
ENCODING_PLANE_NAMES = {
    SIMPLE_18: PLANE_NAMES,
    LC0_STATIC_112: LC0_STATIC_PLANE_NAMES,
    LC0_BT4_112: LC0_BT4_PLANE_NAMES,
}


def available_encodings() -> list[str]:
    return sorted(ENCODING_PLANE_NAMES)


def encoding_num_planes(encoding: str = SIMPLE_18) -> int:
    if encoding not in ENCODING_PLANE_NAMES:
        raise ValueError(f"Unknown board encoding: {encoding}. Available: {available_encodings()}")
    return len(ENCODING_PLANE_NAMES[encoding])


def _square_to_rank_file(square: int) -> tuple[int, int]:
    return 7 - (square // 8), square % 8


def _square_to_lc0_rank_file(square: int, black_to_move: bool) -> tuple[int, int]:
    rank = square // 8
    file = square % 8
    if black_to_move:
        rank = 7 - rank
    return 7 - rank, file


def fen_to_simple_18(fen: str, dtype=np.float32) -> np.ndarray:
    chess_board = parse_fen(fen)
    features = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=dtype)
    piece_to_plane = {piece: idx for idx, piece in enumerate(PIECE_PLANES)}

    for square, piece in chess_board.piece_map().items():
        rank, file = _square_to_rank_file(square)
        features[piece_to_plane[piece.symbol()], rank, file] = 1.0

    if chess_board.turn:
        features[12, :, :] = 1.0

    if chess_board.has_kingside_castling_rights(True):
        features[13, :, :] = 1.0
    if chess_board.has_queenside_castling_rights(True):
        features[14, :, :] = 1.0
    if chess_board.has_kingside_castling_rights(False):
        features[15, :, :] = 1.0
    if chess_board.has_queenside_castling_rights(False):
        features[16, :, :] = 1.0

    if chess_board.ep_square is not None:
        rank, file = _square_to_rank_file(chess_board.ep_square)
        features[17, rank, file] = 1.0
    return features


def fen_to_lc0_static_112(fen: str, dtype=np.float32) -> np.ndarray:
    """LC0-style 112-plane encoding for single-FEN datasets.

    LC0-style networks normally consume history. This dataset only has a current
    FEN, so history slot 0 contains the current board, history slots 1-7 are
    zeros, and repetition planes are zeros.
    """

    chess_board = parse_fen(fen)
    features = np.zeros((encoding_num_planes(LC0_STATIC_112), 8, 8), dtype=dtype)
    piece_to_offset = {piece: idx for idx, piece in enumerate(PIECE_PLANES)}
    for square, piece in chess_board.piece_map().items():
        rank, file = _square_to_rank_file(square)
        features[piece_to_offset[piece.symbol()], rank, file] = 1.0

    aux_start = LC0_HISTORY_STEPS * LC0_PLANES_PER_STEP
    if chess_board.turn:
        features[aux_start, :, :] = 1.0
    else:
        features[aux_start + 1, :, :] = 1.0
    if chess_board.has_kingside_castling_rights(True):
        features[aux_start + 2, :, :] = 1.0
    if chess_board.has_queenside_castling_rights(True):
        features[aux_start + 3, :, :] = 1.0
    if chess_board.has_kingside_castling_rights(False):
        features[aux_start + 4, :, :] = 1.0
    if chess_board.has_queenside_castling_rights(False):
        features[aux_start + 5, :, :] = 1.0
    if chess_board.ep_square is not None:
        rank, file = _square_to_rank_file(chess_board.ep_square)
        features[aux_start + 6, rank, file] = 1.0
    features[aux_start + 7, :, :] = min(chess_board.fullmove_number, 100) / 100.0
    return features


def fen_to_lc0_bt4_112(fen: str, dtype=np.float32) -> np.ndarray:
    """LC0 BT4-style 112-plane encoding for current-FEN records.

    This follows the LC0 112-plane canonical/hectoplies auxiliary layout used
    by modern BT4-style nets. The source data currently has a single FEN per
    record, so only history slot 0 is populated. Older history and repetition
    planes remain zero until the exporter provides game history. This is meant
    for training-from-scratch benchmarks, not loading existing LC0 weights.
    """

    chess_board = parse_fen(fen)
    features = np.zeros((encoding_num_planes(LC0_BT4_112), 8, 8), dtype=dtype)
    black_to_move = not bool(chess_board.turn)
    our_color = bool(chess_board.turn)
    piece_type_to_offset = {
        1: 0,  # pawn
        2: 1,  # knight
        3: 2,  # bishop
        4: 3,  # rook
        5: 4,  # queen
        6: 5,  # king
    }

    for square, piece in chess_board.piece_map().items():
        rank, file = _square_to_lc0_rank_file(square, black_to_move=black_to_move)
        side_offset = 0 if bool(piece.color) == our_color else 6
        features[side_offset + piece_type_to_offset[piece.piece_type], rank, file] = 1.0

    aux_start = LC0_HISTORY_STEPS * LC0_PLANES_PER_STEP
    castling_rooks = [
        (chess_board.has_queenside_castling_rights(True), 0, 0),
        (chess_board.has_queenside_castling_rights(False), 0, 56),
        (chess_board.has_kingside_castling_rights(True), 1, 7),
        (chess_board.has_kingside_castling_rights(False), 1, 63),
    ]
    for has_right, plane_offset, square in castling_rooks:
        if has_right:
            rank, file = _square_to_lc0_rank_file(square, black_to_move=black_to_move)
            features[aux_start + plane_offset, rank, file] = 1.0

    if chess_board.ep_square is not None:
        ep_file = chess_board.ep_square % 8
        features[aux_start + 4, 0, ep_file] = 1.0

    features[aux_start + 5, :, :] = min(max(chess_board.halfmove_clock, 0), 100) / 100.0
    features[aux_start + 7, :, :] = 1.0
    return features


def fen_to_tensor(fen: str, dtype=np.float32, encoding: str = SIMPLE_18) -> np.ndarray:
    if encoding == SIMPLE_18:
        return fen_to_simple_18(fen, dtype=dtype)
    if encoding == LC0_STATIC_112:
        return fen_to_lc0_static_112(fen, dtype=dtype)
    if encoding == LC0_BT4_112:
        return fen_to_lc0_bt4_112(fen, dtype=dtype)
    raise ValueError(f"Unknown board encoding: {encoding}. Available: {available_encodings()}")
