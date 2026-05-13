"""Focused tests for the p027 Sparse Legal-Move Router Head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.sparse_legal_move_router_head import (
    ALLOWED_ABLATIONS,
    SparseLegalMoveRouterHead,
    _piece_type_per_square,
    build_sparse_legal_move_router_head_from_config,
    compute_legal_move_adjacency,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "sparse_legal_move_router_head"
IDEA_DIR = Path("ideas/registry/p027_sparse_legal_move_router_head")
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KNIGHT_ONLY_FEN = "4k3/8/8/8/3N4/8/8/4K3 w - - 0 1"  # white knight on d4
KING_ONLY_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


def _small_config(**overrides):
    base = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 24,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "square_embed_dim": 16,
        "attn_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "gate_init": -1.5,
    }
    base.update(overrides)
    return base


def _fen_to_tensor(fen: str) -> torch.Tensor:
    return torch.from_numpy(fen_to_simple_18(fen)).unsqueeze(0)


def test_model_registered_with_expected_key():
    assert REGISTRY_KEY in available_models()
    model = build_model(REGISTRY_KEY, _small_config())
    assert isinstance(model, SparseLegalMoveRouterHead)


def test_compute_legal_move_adjacency_initial_position():
    board = _fen_to_tensor(INITIAL_FEN)
    adjacency = compute_legal_move_adjacency(board)
    assert adjacency.shape == (1, 64, 64)
    # The initial position has 20 legal moves for white. The aggregated
    # adjacency may count more edges because we treat the per-piece-type
    # attack mask uniformly (pawn captures + forward are both included
    # in the attack table). What we check here is that the number is
    # finite and the mask is non-trivial.
    edges = float(adjacency.sum().item())
    assert edges > 0
    assert torch.all((adjacency == 0) | (adjacency == 1))


def test_compute_legal_move_adjacency_king_only():
    board = _fen_to_tensor(KING_ONLY_FEN)
    adjacency = compute_legal_move_adjacency(board)
    # White king on e1 (rank 1, file e). simple_18 plane row index = 7 (rank 1),
    # col = 4. Its squares it can attack are 5 (rank 1 d/f + rank 2 d/e/f).
    # rank 1 = plane row 7, rank 2 = plane row 6.
    # king on (row 7, col 4) -> source idx 7*8 + 4 = 60.
    src_idx = 7 * 8 + 4
    row = adjacency[0, src_idx]
    assert float(row.sum().item()) == 5.0


def test_compute_legal_move_adjacency_knight():
    board = _fen_to_tensor(KNIGHT_ONLY_FEN)
    adjacency = compute_legal_move_adjacency(board)
    # Knight on d4 -> rank 4 = plane row 4, col d = 3. src_idx = 4*8 + 3 = 35.
    src_idx = 4 * 8 + 3
    row = adjacency[0, src_idx]
    # Knight has 8 legal targets from d4 (all on-board).
    assert float(row.sum().item()) == 8.0


def test_piece_type_per_square_initial_position():
    board = _fen_to_tensor(INITIAL_FEN)
    pieces = _piece_type_per_square(board)
    # Plane row 7 (rank 1) is white pieces: R, N, B, Q, K, B, N, R.
    # In simple_18 piece order (P=0, N=1, B=2, R=3, Q=4, K=5).
    assert int(pieces[0, 56].item()) == 3  # rook a1
    assert int(pieces[0, 57].item()) == 1  # knight b1
    assert int(pieces[0, 58].item()) == 2  # bishop c1
    assert int(pieces[0, 59].item()) == 4  # queen d1
    assert int(pieces[0, 60].item()) == 5  # king e1


def test_forward_shape_and_diagnostics():
    model = build_model(REGISTRY_KEY, _small_config()).eval()
    boards = torch.cat([_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(KNIGHT_ONLY_FEN)], dim=0)
    with torch.no_grad():
        out = model(boards)
    expected = {
        "logits",
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "primitive_gate_entropy",
        "slmr_legal_move_edges",
        "slmr_active_sources",
        "slmr_attention_entropy",
        "slmr_routed_feature_norm",
    }
    missing = expected - set(out.keys())
    assert not missing, f"Missing keys: {sorted(missing)}"
    assert out["logits"].shape == (2,)


def test_zero_delta_recovers_base_logit():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_delta")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["logits"], out["base_logit"])


def test_full_mask_ablation_routes_over_64_squares():
    model = build_model(REGISTRY_KEY, _small_config(ablation="full_64x64_mask")).eval()
    boards = _fen_to_tensor(KING_ONLY_FEN)
    with torch.no_grad():
        out = model(boards)
    # full_64x64_mask should set every source to have 64 targets.
    assert float(out["slmr_legal_move_edges"].item()) == 64.0 * 64.0
    assert torch.isfinite(out["logits"]).all()


def test_self_loop_only_ablation_keeps_logits_finite():
    model = build_model(REGISTRY_KEY, _small_config(ablation="self_loop_only")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    # 64 self-loops per sample.
    assert float(out["slmr_legal_move_edges"].item()) == 64.0


def test_zero_router_features_zeros_delta():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_router_features")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


def test_backward_routes_gradients():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config()).train()
    boards = torch.cat([_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(KNIGHT_ONLY_FEN)], dim=0)
    target = torch.tensor([1.0, 0.0])
    out = model(boards)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_allowed_ablations_match_expected_set():
    expected = {
        "none",
        "zero_delta",
        "disable_gate",
        "trunk_only",
        "full_64x64_mask",
        "self_loop_only",
        "shuffle_adjacency",
        "zero_router_features",
    }
    assert set(ALLOWED_ABLATIONS) == expected


def test_config_yaml_loads_correctly():
    config_path = IDEA_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["idea_id"] == "p027"
    assert config["mode"] == "puzzle_binary"
    assert config["device"] == "nvidia"
    assert config["model"]["name"] == REGISTRY_KEY


def test_rejects_non_simple_18_input():
    with pytest.raises(ValueError):
        SparseLegalMoveRouterHead(input_channels=12)


def test_builder_accepts_aliased_keys():
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "square_embed_dim": 16,
        "attn_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_sparse_legal_move_router_head_from_config(cfg)
    assert isinstance(model, SparseLegalMoveRouterHead)
    assert model.trunk.channels == 24
