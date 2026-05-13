"""Focused tests for the p030 Ray-Parallel SSM Head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.ray_parallel_ssm_head import (
    ALLOWED_ABLATIONS,
    RayParallelSSMHead,
    build_ray_parallel_ssm_head_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "ray_parallel_ssm_head"
IDEA_DIR = Path("ideas/registry/p030_ray_parallel_ssm_head")
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BACK_RANK_MATE_FEN = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"


def _small_config(**overrides):
    base = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 24,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "feature_dim": 4,
        "max_ray_length": 4,
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
    assert isinstance(model, RayParallelSSMHead)


def test_forward_shape_and_diagnostics():
    model = build_model(REGISTRY_KEY, _small_config()).eval()
    boards = torch.cat([_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(BACK_RANK_MATE_FEN)], dim=0)
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
        "ray_ssm_mean_A",
        "ray_ssm_mean_B",
        "ray_ssm_dir_energy_mean",
        "ray_ssm_dir_energy_max",
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


def test_disable_selective_A_keeps_logits_finite():
    model = build_model(REGISTRY_KEY, _small_config(ablation="disable_selective_A")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    # disable_selective_A forces A to a constant 0.5 across all entries.
    assert float(out["ray_ssm_mean_A"].mean().item()) == pytest.approx(0.5, abs=1.0e-5)


def test_disable_selective_B_keeps_logits_finite():
    model = build_model(REGISTRY_KEY, _small_config(ablation="disable_selective_B")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    assert float(out["ray_ssm_mean_B"].mean().item()) == pytest.approx(0.5, abs=1.0e-5)


def test_no_directional_C_keeps_logits_finite():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config(ablation="no_directional_C")).eval()
    boards = _fen_to_tensor(BACK_RANK_MATE_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()


def test_zero_ssm_features_zeros_delta():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_ssm_features")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


def test_disable_gate_keeps_output_gate_at_one():
    model = build_model(REGISTRY_KEY, _small_config(ablation="disable_gate")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_backward_routes_gradients_through_C_param():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config()).train()
    boards = torch.cat([_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(BACK_RANK_MATE_FEN)], dim=0)
    target = torch.tensor([1.0, 0.0])
    out = model(boards)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], target)
    loss.backward()
    # The per-direction C parameter should receive gradient.
    assert model.C_param.grad is not None
    assert model.C_param.grad.abs().sum() > 0


def test_allowed_ablations_match_expected_set():
    expected = {
        "none",
        "zero_delta",
        "disable_gate",
        "trunk_only",
        "disable_selective_A",
        "disable_selective_B",
        "no_directional_C",
        "zero_ssm_features",
    }
    assert set(ALLOWED_ABLATIONS) == expected


def test_config_yaml_loads_correctly():
    config_path = IDEA_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["idea_id"] == "p030"
    assert config["mode"] == "puzzle_binary"
    assert config["device"] == "nvidia"
    assert config["model"]["name"] == REGISTRY_KEY


def test_rejects_non_simple_18_input():
    with pytest.raises(ValueError):
        RayParallelSSMHead(input_channels=12)


def test_rejects_invalid_max_ray_length():
    with pytest.raises(ValueError):
        RayParallelSSMHead(max_ray_length=0)
    with pytest.raises(ValueError):
        RayParallelSSMHead(max_ray_length=8)


def test_builder_accepts_aliased_keys():
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "feature_dim": 4,
        "max_ray_length": 3,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_ray_parallel_ssm_head_from_config(cfg)
    assert isinstance(model, RayParallelSSMHead)
    assert model.trunk.channels == 24
