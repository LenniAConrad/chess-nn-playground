from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


PIECE_VALUES = {
    "p": 1,
    "n": 3,
    "b": 3,
    "r": 5,
    "q": 9,
    "k": 0,
}


def _chess():
    try:
        import chess

        return chess
    except Exception as exc:
        raise ImportError(
            "python-chess is required for FEN validation and board features. "
            "Install requirements.txt first."
        ) from exc


@dataclass(frozen=True)
class FenSummary:
    normalized_fen: str
    side_to_move: str
    piece_count: int
    legal_move_count: int
    is_check: bool
    material_white: int
    material_black: int
    material_balance: int
    board_hash: str


def parse_fen(fen: str):
    chess = _chess()
    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("FEN must be a non-empty string")
    return chess.Board(fen.strip())


def validate_fen(fen: str) -> tuple[bool, str | None]:
    try:
        parse_fen(fen)
        return True, None
    except Exception as exc:
        return False, str(exc)


def normalize_fen(fen: str) -> str:
    board = parse_fen(fen)
    return board.fen()


def side_to_move(fen: str) -> str:
    board = parse_fen(fen)
    return "w" if board.turn else "b"


def count_pieces(fen: str) -> int:
    board = parse_fen(fen)
    return len(board.piece_map())


def count_legal_moves(fen: str) -> int:
    board = parse_fen(fen)
    return board.legal_moves.count()


def is_check(fen: str) -> bool:
    board = parse_fen(fen)
    return bool(board.is_check())


def material_counts(fen: str) -> dict[str, int]:
    board = parse_fen(fen)
    white = 0
    black = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.symbol().lower()]
        if piece.color:
            white += value
        else:
            black += value
    return {"white": white, "black": black, "balance": white - black}


def board_hash_key(fen: str) -> str:
    normalized = normalize_fen(fen)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def summarize_fen(fen: str) -> FenSummary:
    normalized = normalize_fen(fen)
    material = material_counts(normalized)
    return FenSummary(
        normalized_fen=normalized,
        side_to_move=side_to_move(normalized),
        piece_count=count_pieces(normalized),
        legal_move_count=count_legal_moves(normalized),
        is_check=is_check(normalized),
        material_white=material["white"],
        material_black=material["black"],
        material_balance=material["balance"],
        board_hash=board_hash_key(normalized),
    )


def summarize_fen_dict(fen: str) -> dict[str, Any]:
    summary = summarize_fen(fen)
    return {
        "side_to_move": summary.side_to_move,
        "piece_count": summary.piece_count,
        "legal_move_count": summary.legal_move_count,
        "is_check": summary.is_check,
        "material_white": summary.material_white,
        "material_black": summary.material_black,
        "material_balance": summary.material_balance,
        "board_hash": summary.board_hash,
    }
