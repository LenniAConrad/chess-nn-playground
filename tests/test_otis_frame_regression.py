"""Pin the OTIS tensor frame end-to-end so the pawn orientation cannot silently
invert again (it has flipped twice; see URGENT.md).

Frame facts pinned here:
  - board_features encoders write plane row = 7 - (square // 8): flat index 0 is
    a8 and 63 is h1, so the mover's home is internal rank 7 after side-to-move
    canonicalization and mover pawns attack toward LOWER internal rank.
  - Black-to-move canonicalization is a rank-only vertical flip plus color swap,
    i.e. exactly python-chess Board.mirror().
"""
from __future__ import annotations

import chess
import torch

from chess_nn_playground.data.board_features import fen_to_tensor
from chess_nn_playground.models.trunk.oriented_tactical_sheaf import (
    BoardStateAdapter,
    TacticalIncidenceBuilder,
    _square_coordinates,
)


def _flat(square_name: str) -> int:
    square = chess.parse_square(square_name)
    return (7 - square // 8) * 8 + square % 8


def _incidence(fen: str, encoding: str = "simple_18", channels: int = 18):
    x = torch.from_numpy(fen_to_tensor(fen, encoding=encoding)).float().unsqueeze(0)
    adapter = BoardStateAdapter(input_channels=channels, encoding=encoding)
    state = adapter(x)
    incidence = TacticalIncidenceBuilder()(state.piece_state, state.occupancy)
    return state, incidence


def test_mover_pawn_attacks_point_forward_in_encoder_frame():
    _, incidence = _incidence(chess.STARTING_FEN)
    e2, d3, f3 = _flat("e2"), _flat("d3"), _flat("f3")
    attacks = torch.nonzero(incidence.our_attack[0, e2] > 0.5).flatten().tolist()
    assert sorted(attacks) == sorted([d3, f3]), (
        f"e2 pawn (flat {e2}) must attack d3/f3 (flat {d3}/{f3}), got {attacks}"
    )
    e7, d6, f6 = _flat("e7"), _flat("d6"), _flat("f6")
    attacks = torch.nonzero(incidence.them_attack[0, e7] > 0.5).flatten().tolist()
    assert sorted(attacks) == sorted([d6, f6])
    # relation channel 10 (pawn_attack_forward_oriented) carries both directions
    assert incidence.relation_masks[0, 10, e2, d3] == 1.0
    assert incidence.relation_masks[0, 10, e7, f6] == 1.0


def test_pawn_attack_reaches_enemy_piece_relation():
    # After 1.e4 d5 the e4 pawn attacks the d5 pawn: us_attacks_them must fire.
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    _, incidence = _incidence(fen)
    assert incidence.relation_masks[0, 0, _flat("e4"), _flat("d5")] == 1.0
    assert incidence.relation_masks[0, 1, _flat("d5"), _flat("e4")] == 1.0


def test_black_to_move_canonicalization_equals_python_chess_mirror():
    # Black pawn on d4 makes the e3 en-passant capture legal, so the ep field
    # survives python-chess fen() normalization on both sides of the mirror.
    fen = "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3"
    mirrored = chess.Board(fen).mirror().fen()
    adapter = BoardStateAdapter(input_channels=18, encoding="simple_18")
    xb = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0)
    xm = torch.from_numpy(fen_to_tensor(mirrored)).float().unsqueeze(0)
    sb, sm = adapter(xb), adapter(xm)
    assert torch.equal(sb.piece_state, sm.piece_state)
    assert torch.equal(sb.square_raw, sm.square_raw)


def test_lc0_static_castling_channels_follow_mover():
    # Only black has a kingside right; with black to move it must land in the
    # mover ("white") kingside channel 106 after canonicalization.
    fen = "r3k2r/8/8/8/8/8/8/R3K2R b k - 0 1"
    state, _ = _incidence(fen, encoding="lc0_static_112", channels=112)
    assert state.square_raw[0, :, 106].sum() == 64.0
    assert state.square_raw[0, :, 108].sum() == 0.0


def test_promotion_distance_is_zero_at_mover_promotion_rank():
    coords = _square_coordinates()
    rank = (torch.arange(64) // 8).float()
    assert torch.allclose(coords[:, 5], rank / 7.0)
