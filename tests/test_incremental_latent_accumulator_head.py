"""Focused tests for the p028 Incremental Latent Accumulator Head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.incremental_latent_accumulator_head import (
    ALLOWED_ABLATIONS,
    IncrementalLatentAccumulatorHead,
    _own_king_square,
    build_incremental_latent_accumulator_head_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "incremental_latent_accumulator_head"
IDEA_DIR = Path("ideas/registry/p028_incremental_latent_accumulator_head")
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
WHITE_KING_E1_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
BLACK_KING_TO_MOVE_FEN = "4k3/8/8/8/8/8/8/4K3 b - - 0 1"


def _small_config(**overrides):
    base = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 24,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "global_dim": 12,
        "king_dim": 4,
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
    assert isinstance(model, IncrementalLatentAccumulatorHead)


def test_own_king_square_returns_white_king_when_white_to_move():
    board = _fen_to_tensor(WHITE_KING_E1_FEN)
    king = _own_king_square(board)
    # White king on e1: plane row 7 (rank 1), col 4 (e-file) -> 7*8 + 4 = 60.
    assert int(king.item()) == 60


def test_own_king_square_returns_black_king_when_black_to_move():
    board = _fen_to_tensor(BLACK_KING_TO_MOVE_FEN)
    king = _own_king_square(board)
    # Black king on e8: plane row 0 (rank 8), col 4 -> 0*8 + 4 = 4.
    assert int(king.item()) == 4


def test_forward_shape_and_diagnostics():
    model = build_model(REGISTRY_KEY, _small_config()).eval()
    boards = torch.cat([_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(WHITE_KING_E1_FEN)], dim=0)
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
        "ila_global_norm",
        "ila_king_norm",
        "ila_latent_norm",
        "ila_active_cells",
        "ila_king_index",
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


def test_zero_king_accumulator_keeps_logits_finite():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_king_accumulator")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    assert float(out["ila_king_norm"].item()) == 0.0


def test_zero_global_accumulator_keeps_logits_finite():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_global_accumulator")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    assert float(out["ila_global_norm"].item()) == 0.0


def test_linear_only_ablation_skips_phi():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config(ablation="linear_only")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()


def test_disable_gate_keeps_gate_at_one():
    model = build_model(REGISTRY_KEY, _small_config(ablation="disable_gate")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_active_cells_initial_position():
    model = build_model(REGISTRY_KEY, _small_config()).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert float(out["ila_active_cells"].item()) == 32.0


def test_backward_routes_gradients_through_both_embedding_tables():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config()).train()
    boards = torch.cat([_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(WHITE_KING_E1_FEN)], dim=0)
    target = torch.tensor([1.0, 0.0])
    out = model(boards)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], target)
    loss.backward()
    assert model.global_embedding.grad is not None
    assert model.global_embedding.grad.abs().sum() > 0
    assert model.king_embedding.grad is not None
    assert model.king_embedding.grad.abs().sum() > 0


def test_allowed_ablations_match_expected_set():
    expected = {
        "none",
        "zero_delta",
        "disable_gate",
        "trunk_only",
        "zero_global_accumulator",
        "zero_king_accumulator",
        "linear_only",
        "shuffle_square_order",
    }
    assert set(ALLOWED_ABLATIONS) == expected


def test_config_yaml_loads_correctly():
    config_path = IDEA_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["idea_id"] == "p028"
    assert config["mode"] == "puzzle_binary"
    assert config["device"] == "nvidia"
    assert config["model"]["name"] == REGISTRY_KEY


def test_rejects_non_simple_18_input():
    with pytest.raises(ValueError):
        IncrementalLatentAccumulatorHead(input_channels=12)


def test_rejects_invalid_dims():
    with pytest.raises(ValueError):
        IncrementalLatentAccumulatorHead(global_dim=0)
    with pytest.raises(ValueError):
        IncrementalLatentAccumulatorHead(king_dim=0)


def test_builder_accepts_aliased_keys():
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "global_dim": 12,
        "king_dim": 4,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_incremental_latent_accumulator_head_from_config(cfg)
    assert isinstance(model, IncrementalLatentAccumulatorHead)
    assert model.trunk.channels == 24
