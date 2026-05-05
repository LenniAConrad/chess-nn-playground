from __future__ import annotations

import torch

from chess_nn_playground.models.gpt_research_architectures import CompactBoardEncoder
from chess_nn_playground.models.gpt_research_architectures import DeterministicTacticalMaskBuilder
from chess_nn_playground.models.research_packet_probe import _piece_occupancy


def test_compact_board_encoder_material_delta_follows_simple18_side_to_move():
    encoder = CompactBoardEncoder(input_channels=18, channels=4, depth=1, hidden_dim=8, use_batchnorm=False)
    x = torch.zeros(2, 18, 8, 8)
    x[:, 0, 0, 0] = 1.0
    x[:, 1, 0, 1] = 1.0
    x[:, 6, 7, 7] = 1.0
    x[0, 12] = 1.0
    x[1, 12] = 0.0

    stats = encoder.board_stats(x)

    assert stats[0, 15] > 0.0
    assert stats[1, 15] < 0.0


def test_tactical_masks_treat_black_pieces_as_own_when_black_to_move():
    masks = DeterministicTacticalMaskBuilder()
    x = torch.zeros(1, 18, 8, 8)
    x[:, 12] = 0.0
    x[0, 6, 3, 3] = 1.0
    x[0, 0, 4, 4] = 1.0

    out = masks(x)

    assert out[0, 5, 3, 4] > 0.0
    assert out[0, 6, 3, 4] == 0.0


def test_research_packet_occupancy_treats_black_as_own_when_black_to_move():
    x = torch.zeros(1, 18, 8, 8)
    x[:, 12] = 0.0
    x[0, 6, 3, 3] = 1.0
    x[0, 0, 4, 4] = 1.0

    _piece, own, opp, occupancy, empty = _piece_occupancy(x)

    assert own[0, 0, 3, 3] == 1.0
    assert own[0, 0, 4, 4] == 0.0
    assert opp[0, 0, 4, 4] == 1.0
    assert occupancy[0, 0, 3, 3] == 1.0
    assert empty[0, 0, 0, 0] == 1.0
