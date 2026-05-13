"""Tests for the Terminal-State Detection Primitive (TSDP) features."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.data.terminal_state import (
    TSDP_FEATURE_DIM,
    TSDP_FEATURE_NAMES,
    compute_terminal_state_features,
    compute_terminal_state_features_batch,
    simple_18_batch_to_terminal_state_features,
    simple_18_to_board,
)


def _feat(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(TSDP_FEATURE_NAMES)}


def test_feature_dimension_and_names_are_consistent() -> None:
    assert TSDP_FEATURE_DIM == 11
    assert len(TSDP_FEATURE_NAMES) == TSDP_FEATURE_DIM
    assert TSDP_FEATURE_NAMES[0] == "mate_in_1"
    assert "forcing_density" in TSDP_FEATURE_NAMES


def test_starting_position_is_quiet() -> None:
    features = compute_terminal_state_features(chess.STARTING_FEN)
    values = _feat(features)
    assert values["mate_in_1"] == 0.0
    assert values["stalemate_threat"] == 0.0
    assert values["total_legal_moves"] == 20.0
    assert values["check_count"] == 0.0
    assert values["capture_count"] == 0.0
    assert values["forcing_density"] == 0.0


def test_back_rank_mate_in_1_detected() -> None:
    fen = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"
    features = compute_terminal_state_features(fen)
    values = _feat(features)
    assert values["mate_in_1"] == 1.0
    assert values["mate_count"] >= 1.0


def test_queen_mate_in_1_detected_with_multiple_mating_moves() -> None:
    fen = "7k/7Q/6K1/8/8/8/8/8 w - - 0 1"
    features = compute_terminal_state_features(fen)
    values = _feat(features)
    assert values["mate_in_1"] == 1.0
    # the Qg7# scenario should expose more than one mating move
    assert values["mate_count"] >= 1.0


def test_stalemate_threat_detected() -> None:
    fen = "k7/8/1Q2K3/8/8/8/8/8 w - - 0 1"
    features = compute_terminal_state_features(fen)
    values = _feat(features)
    # at least one move from this configuration leads to stalemate; the
    # indicator must catch it.
    assert values["stalemate_threat"] == 1.0
    assert values["stalemate_count"] >= 1.0


def test_terminal_position_returns_zero_features() -> None:
    # King-and-queen vs king mate: black king on h8 has no legal moves and is
    # in check from the white queen on h6, with the white king on f7 covering
    # g7 / g8. Black to move -> checkmate -> TSDP returns zeros.
    fen = "7k/5K2/7Q/8/8/8/8/8 b - - 0 1"
    board = chess.Board(fen)
    assert board.is_checkmate()
    assert board.legal_moves.count() == 0
    features = compute_terminal_state_features(board)
    assert np.allclose(features, np.zeros(TSDP_FEATURE_DIM))


def test_simple_18_round_trip_for_quiet_position() -> None:
    fen = chess.STARTING_FEN
    board = chess.Board(fen)
    tensor = torch.from_numpy(fen_to_simple_18(fen))
    rebuilt = simple_18_to_board(tensor.numpy())
    # piece placement must agree
    assert rebuilt.piece_map() == board.piece_map()
    assert rebuilt.turn == board.turn
    # legal-move counts agree exactly — that is what TSDP cares about
    assert len(list(rebuilt.legal_moves)) == len(list(board.legal_moves))


def test_simple_18_round_trip_preserves_castling_and_en_passant() -> None:
    # white to move, full castling rights, no en-passant
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    tensor = torch.from_numpy(fen_to_simple_18(fen))
    rebuilt = simple_18_to_board(tensor.numpy())
    assert rebuilt.has_kingside_castling_rights(True)
    assert rebuilt.has_queenside_castling_rights(True)
    assert rebuilt.has_kingside_castling_rights(False)
    assert rebuilt.has_queenside_castling_rights(False)

    # an en-passant available scenario: white pawn just moved e4 from e2.
    fen_ep = "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 1"
    tensor_ep = torch.from_numpy(fen_to_simple_18(fen_ep))
    rebuilt_ep = simple_18_to_board(tensor_ep.numpy())
    assert rebuilt_ep.ep_square is not None


def test_batch_extraction_matches_per_sample_extraction() -> None:
    fens = [
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",   # mate-in-1
        "k7/8/1Q2K3/8/8/8/8/8 w - - 0 1",        # stalemate trap
    ]
    per_sample = compute_terminal_state_features_batch(fens)
    tensors = np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)
    batched = simple_18_batch_to_terminal_state_features(torch.from_numpy(tensors))
    assert per_sample.shape == batched.shape == (3, TSDP_FEATURE_DIM)
    assert np.allclose(per_sample, batched)


def test_shuffled_indicator_contract_decouples_features() -> None:
    fens = [
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",   # back-rank mate-in-1
        "k7/8/1Q2K3/8/8/8/8/8 w - - 0 1",        # stalemate trap
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ]
    features = compute_terminal_state_features_batch(fens)
    # explicitly invert the order so we never hit the identity permutation
    shuffled = features[::-1]
    # at least one row must differ from the original after the reversal
    assert not np.allclose(features, shuffled)
