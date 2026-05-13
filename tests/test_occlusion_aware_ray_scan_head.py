"""Focused tests for the p029 Occlusion-Aware Ray Scan Head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.occlusion_aware_ray_scan_head import (
    ALLOWED_ABLATIONS,
    OcclusionAwareRayScanHead,
    build_occlusion_aware_ray_scan_head_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "occlusion_aware_ray_scan_head"
IDEA_DIR = Path("ideas/registry/p029_occlusion_aware_ray_scan_head")
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
    assert isinstance(model, OcclusionAwareRayScanHead)


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
        "oars_mean_blocker_gate",
        "oars_dir_energy_mean",
        "oars_dir_energy_max",
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


def test_disable_blocker_gate_keeps_logits_finite():
    model = build_model(REGISTRY_KEY, _small_config(ablation="disable_blocker_gate")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    # With the blocker gate forced to 1, the average blocker gate should
    # be exactly 1.0 (the diagnostic is computed from the gate value
    # *before* the disable-blocker substitution would override it, so
    # this checks the model honours the contract).
    # Note: we report the diagnostic from gate_value, which gets
    # overwritten by ones in disable_blocker_gate, so 1.0 is expected.
    assert float(out["oars_mean_blocker_gate"].item()) == pytest.approx(1.0, abs=1.0e-6)


def test_shuffle_directions_keeps_logits_finite():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config(ablation="shuffle_directions")).eval()
    boards = _fen_to_tensor(BACK_RANK_MATE_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()


def test_zero_oars_features_zeros_delta():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_oars_features")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


def test_disable_gate_holds_output_gate_at_one():
    model = build_model(REGISTRY_KEY, _small_config(ablation="disable_gate")).eval()
    boards = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_backward_routes_gradients():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config()).train()
    boards = torch.cat([_fen_to_tensor(INITIAL_FEN), _fen_to_tensor(BACK_RANK_MATE_FEN)], dim=0)
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
        "disable_blocker_gate",
        "shuffle_directions",
        "zero_oars_features",
    }
    assert set(ALLOWED_ABLATIONS) == expected


def test_config_yaml_loads_correctly():
    config_path = IDEA_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["idea_id"] == "p029"
    assert config["mode"] == "puzzle_binary"
    assert config["device"] == "nvidia"
    assert config["model"]["name"] == REGISTRY_KEY


def test_rejects_non_simple_18_input():
    with pytest.raises(ValueError):
        OcclusionAwareRayScanHead(input_channels=12)


def test_rejects_invalid_max_ray_length():
    with pytest.raises(ValueError):
        OcclusionAwareRayScanHead(max_ray_length=0)
    with pytest.raises(ValueError):
        OcclusionAwareRayScanHead(max_ray_length=8)


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
    model = build_occlusion_aware_ray_scan_head_from_config(cfg)
    assert isinstance(model, OcclusionAwareRayScanHead)
    assert model.trunk.channels == 24
