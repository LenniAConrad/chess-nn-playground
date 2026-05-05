from __future__ import annotations

from chess_nn_playground.data.fen_utils import normalize_fen, summarize_fen_dict, validate_fen


def test_valid_fen_normalizes():
    fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    assert validate_fen(fen)[0]
    assert normalize_fen(fen) == fen
    summary = summarize_fen_dict(fen)
    assert summary["side_to_move"] == "w"
    assert summary["piece_count"] == 2


def test_invalid_fen_rejected():
    valid, error = validate_fen("not a fen")
    assert not valid
    assert error
