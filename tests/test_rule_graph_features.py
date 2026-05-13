"""Tests for the shared rule-graph helper module used by p006-p011."""
from __future__ import annotations

import numpy as np
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.rule_graph_features import (
    MAX_RAY_LEN,
    NUM_DIRECTIONS,
    NUM_PIECE_TYPES,
    SQUARES,
    SquareTokenEmbedder,
    compute_attack_relations,
    compute_legal_move_graph,
    compute_ray_transmittance,
    first_blocker_indices,
    occupancy_from_board,
    piece_planes_flat,
    rule_geometry,
    side_to_move_from_board,
)


INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
EMPTY_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
TACTICAL_FEN = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"


def _boards(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(f) for f in fens])).float()


def test_geometry_table_shapes() -> None:
    geom = rule_geometry()
    assert geom.geom_attacks.shape == (NUM_PIECE_TYPES, 2, SQUARES, SQUARES)
    assert geom.between.shape == (SQUARES, SQUARES, SQUARES)
    assert geom.ray_step_target.shape == (SQUARES, NUM_DIRECTIONS, MAX_RAY_LEN)
    assert geom.ray_step_valid.shape == (SQUARES, NUM_DIRECTIONS, MAX_RAY_LEN)
    assert geom.ray_step_count.shape == (SQUARES, NUM_DIRECTIONS)


def test_occupancy_and_stm_from_initial() -> None:
    boards = _boards([INITIAL_FEN])
    occ = occupancy_from_board(boards)
    assert occ.shape == (1, 64)
    # Initial position: 32 pieces, all in [0, 1].
    assert occ.sum().item() == 32.0
    stm = side_to_move_from_board(boards)
    assert stm.item() == 1.0  # white to move


def test_attack_relations_initial_position() -> None:
    geom = rule_geometry()
    boards = _boards([INITIAL_FEN])
    attacks, rays = compute_attack_relations(boards, geom)
    assert attacks.shape == (1, 2, 64, 64)
    assert rays.shape == (1, 2, 64, 64)
    # Initial position: 32 pawns attack 14 each side. + knights, etc.
    # Just sanity-check positivity and shape; exact count is tested via the
    # legal-graph test downstream.
    assert attacks.sum().item() > 0.0


def test_legal_move_graph_initial_edge_count() -> None:
    """Initial position attack-derived legal graph: 14 pawn attacks + 4
    knight attacks for white-to-move = 18 edges."""
    geom = rule_geometry()
    boards = _boards([INITIAL_FEN])
    legal = compute_legal_move_graph(boards, geom)
    assert legal.shape == (1, 64, 64)
    assert int(legal.sum().item()) == 18


def test_ray_transmittance_bounds_and_empty_board() -> None:
    geom = rule_geometry()
    boards = _boards([INITIAL_FEN, EMPTY_FEN])
    trans = compute_ray_transmittance(boards, geom)
    assert trans.shape == (2, 64, 8, 7)
    assert torch.all(trans >= 0.0)
    assert torch.all(trans <= 1.0 + 1e-5)
    # The mostly-empty board (only two kings) should be much closer to the
    # ray_valid mask than the initial position because there are fewer
    # blockers along each ray.
    valid_mask = geom.ray_step_valid.to(dtype=trans.dtype)
    delta_empty = (trans[1] - valid_mask).abs().sum().item()
    delta_initial = (trans[0] - valid_mask).abs().sum().item()
    assert delta_empty < delta_initial


def test_first_blocker_indices_on_initial() -> None:
    geom = rule_geometry()
    boards = _boards([INITIAL_FEN])
    target, has_blocker = first_blocker_indices(boards, geom)
    assert target.shape == (1, 64, 8)
    assert has_blocker.shape == (1, 64, 8)
    # On the initial position rays from interior squares hit the pawn ranks
    # quickly, so most rays have a blocker.
    assert int(has_blocker.sum().item()) > 200


def test_square_token_embedder_shapes_and_devices() -> None:
    boards = _boards([INITIAL_FEN, TACTICAL_FEN])
    embedder = SquareTokenEmbedder(input_channels=18, embed_dim=8, hidden_dim=8)
    tokens = embedder(boards)
    assert tokens.shape == (2, 64, 8)
    # Gradient must flow through the embedder.
    loss = tokens.pow(2).sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in embedder.parameters())
