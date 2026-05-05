from __future__ import annotations

import chess

from chess_nn_playground.data.fen_utils import parse_fen


def _clip_count(value: int, cap: int) -> float:
    if cap <= 0:
        return 0.0
    return min(max(float(value), 0.0) / float(cap), 1.0)


def _king_zone_squares(king_square: int | None) -> list[int]:
    if king_square is None:
        return []
    rank = chess.square_rank(king_square)
    file = chess.square_file(king_square)
    squares: list[int] = []
    for rank_delta in (-1, 0, 1):
        for file_delta in (-1, 0, 1):
            zone_rank = rank + rank_delta
            zone_file = file + file_delta
            if 0 <= zone_rank <= 7 and 0 <= zone_file <= 7:
                squares.append(chess.square(zone_file, zone_rank))
    return squares


def tactical_texture_score(fen: str) -> float:
    """Deterministic current-board tactical texture in [0, 1].

    This intentionally uses rule-only facts from the current FEN. It is suitable
    for weighting VetoSelect self-mined decoy targets, but it is not a model
    input and does not read engine, source, or verification metadata.
    """

    board = parse_fen(fen)
    legal_moves = list(board.legal_moves)
    checking_moves = 0
    captures = 0
    promotions = 0
    for move in legal_moves:
        if board.is_capture(move):
            captures += 1
        if move.promotion is not None:
            promotions += 1
        if board.gives_check(move):
            checking_moves += 1

    side = bool(board.turn)
    opponent_king = board.king(not side)
    king_zone_attacks = sum(len(board.attackers(side, square)) for square in _king_zone_squares(opponent_king))

    pinned = 0
    hanging = 0
    side_promotion_pressure = promotions > 0
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.KING and board.is_pinned(piece.color, square):
            pinned += 1
        if piece.piece_type != chess.KING:
            attacked = bool(board.attackers(not piece.color, square))
            defended = bool(board.attackers(piece.color, square))
            if attacked and not defended:
                hanging += 1
        if piece.piece_type == chess.PAWN and piece.color == side:
            rank = chess.square_rank(square)
            side_promotion_pressure = side_promotion_pressure or (piece.color and rank >= 5) or (
                not piece.color and rank <= 2
            )

    raw = (
        0.25 * float(checking_moves > 0)
        + 0.20 * _clip_count(checking_moves, 3)
        + 0.15 * _clip_count(captures, 6)
        + 0.15 * _clip_count(king_zone_attacks, 6)
        + 0.10 * float(pinned > 0)
        + 0.10 * _clip_count(hanging, 4)
        + 0.05 * float(side_promotion_pressure)
    )
    return min(max(raw, 0.0), 1.0)
