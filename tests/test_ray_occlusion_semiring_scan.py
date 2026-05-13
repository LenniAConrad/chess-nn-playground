"""Focused tests for the p010 Ray-Occlusion Semiring Scan primitive."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.ray_occlusion_semiring_scan import (
    ALLOWED_ABLATIONS,
    RayOcclusionSemiringScan,
    build_ray_occlusion_semiring_scan_from_config,
)
from chess_nn_playground.models.primitives.rule_graph_features import (
    compute_ray_transmittance,
    rule_geometry,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "ray_occlusion_semiring_scan"
IDEA_DIR = Path("ideas/registry/p010_ray_occlusion_semiring_scan")

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TACTICAL_FEN = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"
EMPTY_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 24,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "token_embed_dim": 16,
        "ray_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "gate_init": -2.0,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(f) for f in fens])).float()


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, RayOcclusionSemiringScan)


def test_forward_shapes_and_keys() -> None:
    model = RayOcclusionSemiringScan(**_toy_kwargs()).eval()
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    assert out["primitive_gate"].shape == (2,)
    assert out["ros_mean_transmittance"].shape == (2,)
    assert out["ros_step_decay_mean"].shape == (2,)
    assert out["ros_pooled_norm"].shape == (2,)


def test_transmittance_bounds() -> None:
    geom = rule_geometry()
    boards = _board_batch([INITIAL_FEN, EMPTY_FEN])
    trans = compute_ray_transmittance(boards, geom)
    assert torch.all(trans >= 0.0)
    assert torch.all(trans <= 1.0 + 1e-5)


def test_transmittance_higher_on_empty_than_initial() -> None:
    geom = rule_geometry()
    boards = _board_batch([INITIAL_FEN, EMPTY_FEN])
    trans = compute_ray_transmittance(boards, geom)
    assert trans[1].sum().item() > trans[0].sum().item()


def test_zero_delta_recovers_base_logit() -> None:
    model = RayOcclusionSemiringScan(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_uniform_transmittance_collapses_to_ray_validity() -> None:
    model_uniform = RayOcclusionSemiringScan(**_toy_kwargs(), ablation="uniform_transmittance").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model_uniform(boards)
    # Mean transmittance under the uniform ablation collapses to mean ray validity.
    assert out["ros_mean_transmittance"][0].item() == pytest.approx(0.40625, abs=1e-3)


def test_disable_gate_holds_gate_at_one() -> None:
    model = RayOcclusionSemiringScan(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = RayOcclusionSemiringScan(**_toy_kwargs())
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    out = model(boards)
    loss = out["logits"].pow(2).mean()
    loss.backward()
    trunk_param = next(model.trunk.parameters())
    direction_param = model.direction_linears[0].weight
    decay_grad = model.step_decay_logit.grad
    gate_param = next(model.gate_head.parameters())
    assert trunk_param.grad is not None
    assert direction_param.grad is not None
    assert decay_grad is not None
    assert gate_param.grad is not None


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        RayOcclusionSemiringScan(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input_at_init() -> None:
    with pytest.raises(ValueError):
        RayOcclusionSemiringScan(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        RayOcclusionSemiringScan(input_channels=18, num_classes=3)


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "token_embed_dim": 16,
        "ray_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_ray_occlusion_semiring_scan_from_config(cfg)
    assert isinstance(model, RayOcclusionSemiringScan)
    assert model.trunk.channels == 24


def test_idea_config_yaml_matches_registry_key() -> None:
    config_path = IDEA_DIR / "config.yaml"
    if not config_path.exists():
        pytest.skip("Idea config not present in worktree.")
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["idea_id"] == "p010"
    assert set(ALLOWED_ABLATIONS) >= {"none", "uniform_transmittance", "constant_direction"}
