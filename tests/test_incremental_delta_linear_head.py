"""Focused tests for the p025 Incremental Delta-Linear Accumulator Head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.incremental_delta_linear_head import (
    ALLOWED_ABLATIONS,
    IncrementalDeltaLinearHead,
    build_incremental_delta_linear_head_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "incremental_delta_linear_head"
IDEA_DIR = Path("ideas/registry/p025_incremental_delta_linear_head")
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BACK_RANK_MATE_FEN = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"
EMPTY_BOARD_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


def _small_config(**overrides):
    base = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 24,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "accumulator_dim": 12,
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
    assert isinstance(model, IncrementalDeltaLinearHead)


def test_forward_shape_and_diagnostics():
    model = build_model(REGISTRY_KEY, _small_config()).eval()
    boards = torch.cat([_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(BACK_RANK_MATE_FEN)], dim=0)
    with torch.no_grad():
        out = model(boards)
    expected_keys = {
        "logits",
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "primitive_gate_entropy",
        "idl_accumulator_norm",
        "idl_accumulator_state_l2",
        "idl_active_cells",
    }
    missing = expected_keys - set(out.keys())
    assert not missing, f"Missing diagnostics keys: {sorted(missing)}"
    assert out["logits"].shape == (2,)
    assert torch.all(out["primitive_gate"] >= 0.0)
    assert torch.all(out["primitive_gate"] <= 1.0)


def test_empty_board_has_zero_active_cells():
    model = build_model(REGISTRY_KEY, _small_config()).eval()
    boards = _fen_to_tensor(EMPTY_BOARD_FEN)
    with torch.no_grad():
        out = model(boards)
    # Empty-board FEN still has two kings (legal minimum); active_cells = 2.
    assert float(out["idl_active_cells"].item()) == 2.0


def test_initial_position_has_thirty_two_active_cells():
    model = build_model(REGISTRY_KEY, _small_config()).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert float(out["idl_active_cells"].item()) == 32.0


def test_zero_delta_recovers_base_logit():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_delta")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_zeros_delta():
    model = build_model(REGISTRY_KEY, _small_config(ablation="trunk_only")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


def test_zero_accumulator_keeps_logits_finite():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_accumulator")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


def test_disable_gate_holds_gate_at_one():
    model = build_model(REGISTRY_KEY, _small_config(ablation="disable_gate")).eval()
    boards = _fen_to_tensor(BACK_RANK_MATE_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))
    assert torch.allclose(out["primitive_delta"], out["primitive_delta_raw"])


def test_shuffle_squares_keeps_logits_finite():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config(ablation="shuffle_squares")).eval()
    boards = torch.cat(
        [_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(BACK_RANK_MATE_FEN)], dim=0
    )
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    # active_cells is computed on the un-shuffled board, so it should still be
    # 32 and 5 for these two positions.
    assert float(out["idl_active_cells"][0].item()) == 32.0


def test_backward_routes_gradients_through_trunk_and_head():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config()).train()
    boards = torch.cat(
        [_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(BACK_RANK_MATE_FEN)], dim=0
    )
    target = torch.tensor([1.0, 0.0])
    out = model(boards)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], target)
    loss.backward()
    head_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("trunk.") and p.requires_grad
    ]
    trunk_params = [p for p in model.trunk.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head_params)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trunk_params)


def test_allowed_ablations_match_expected_set():
    expected = {
        "none",
        "zero_delta",
        "disable_gate",
        "trunk_only",
        "shuffle_squares",
        "permute_piece_types",
        "zero_accumulator",
    }
    assert set(ALLOWED_ABLATIONS) == expected


def test_config_yaml_loads_and_routes_through_registry():
    config_path = IDEA_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["idea_id"] == "p025"
    assert config["mode"] == "puzzle_binary"
    assert config["device"] == "nvidia"
    assert config["data"]["encoding"] == "simple_18"
    assert config["model"]["name"] == REGISTRY_KEY
    assert config["model"]["num_classes"] == 1
    assert config["model"]["input_channels"] == 18


def test_rejects_non_simple_18_input():
    with pytest.raises(ValueError):
        IncrementalDeltaLinearHead(input_channels=12)


def test_rejects_multi_class_num_classes():
    with pytest.raises(ValueError):
        IncrementalDeltaLinearHead(num_classes=3)


def test_rejects_unknown_ablation():
    with pytest.raises(ValueError):
        IncrementalDeltaLinearHead(ablation="not_a_real_ablation")


def test_builder_accepts_aliased_channel_keys():
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "accumulator_dim": 12,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_incremental_delta_linear_head_from_config(cfg)
    assert isinstance(model, IncrementalDeltaLinearHead)
    assert model.trunk.channels == 24
