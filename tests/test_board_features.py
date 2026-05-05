from __future__ import annotations

import pytest

from chess_nn_playground.data.board_features import (
    LC0_BT4_112,
    NUM_INPUT_PLANES,
    available_encodings,
    encoding_num_planes,
    fen_to_tensor,
)


def test_board_feature_tensor_shape():
    tensor = fen_to_tensor("8/8/8/8/8/8/8/K6k w - - 0 1")
    assert tensor.shape == (NUM_INPUT_PLANES, 8, 8)
    assert tensor.dtype.name == "float32"
    assert tensor.sum() >= 2


def test_lc0_bt4_encoding_shape_and_aux_planes():
    tensor = fen_to_tensor(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        encoding=LC0_BT4_112,
    )
    assert LC0_BT4_112 in available_encodings()
    assert tensor.shape == (encoding_num_planes(LC0_BT4_112), 8, 8)
    assert tensor[0].sum() == 8
    assert tensor[6].sum() == 8
    assert tensor[13:104].sum() == 0
    assert tensor[104, 7, 0] == 1
    assert tensor[104, 0, 0] == 1
    assert tensor[104].sum() == 2
    assert tensor[105, 7, 7] == 1
    assert tensor[105, 0, 7] == 1
    assert tensor[105].sum() == 2
    assert tensor[106].sum() == 0
    assert tensor[107].sum() == 0
    assert tensor[108].sum() == 0
    assert tensor[109].sum() == 0
    assert tensor[110].sum() == 0
    assert tensor[111].sum() == 64


def test_lc0_bt4_encoding_uses_side_to_move_perspective():
    tensor = fen_to_tensor(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        encoding=LC0_BT4_112,
    )
    assert tensor[0, 6].sum() == 8
    assert tensor[5, 7, 4] == 1
    assert tensor[6, 1].sum() == 8
    assert tensor[11, 0, 4] == 1


def test_lc0_bt4_encoding_en_passant_file_and_rule50_plane():
    tensor = fen_to_tensor(
        "rnbqkbnr/ppppp1pp/6p1/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 27 3",
        encoding=LC0_BT4_112,
    )
    assert tensor[108, 0, 5] == 1
    assert tensor[108].sum() == 1
    assert tensor[109, 0, 0] == pytest.approx(0.27)
    assert tensor[109].sum() == pytest.approx(64 * 0.27)
