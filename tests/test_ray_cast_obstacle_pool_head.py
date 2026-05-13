"""Focused tests for the p026 Ray-Cast Obstacle Pooling Head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.ray_cast_obstacle_pool_head import (
    ALLOWED_ABLATIONS,
    NUM_DIRECTIONS,
    RAY_DIRECTIONS,
    RayCastObstaclePoolHead,
    _shift_along_direction,
    build_ray_cast_obstacle_pool_head_from_config,
    occupancy_from_simple_18,
    ray_pool,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "ray_cast_obstacle_pool_head"
IDEA_DIR = Path("ideas/registry/p026_ray_cast_obstacle_pool_head")
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
        "gamma_init": 0.7,
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
    assert isinstance(model, RayCastObstaclePoolHead)


def test_shift_along_direction_zero_pads_outside_board():
    x = torch.arange(64).view(1, 1, 8, 8).float()
    # Convention: _shift(x, row_step, file_step) sets
    # out[:, :, i, j] = x[:, :, i + row_step, j + file_step], with zero
    # padding for indices outside the 8x8 board. So row_step=1 reads
    # one row "later" from the source; the last row of the output (no
    # source available) must be zero and the first 7 rows of output
    # must equal the LAST 7 rows of input.
    out = _shift_along_direction(x, 1, 0)
    assert torch.all(out[0, 0, -1] == 0)
    assert torch.equal(out[0, 0, :-1], x[0, 0, 1:])
    out = _shift_along_direction(x, 0, 1)
    assert torch.all(out[0, 0, :, -1] == 0)
    assert torch.equal(out[0, 0, :, :-1], x[0, 0, :, 1:])


def test_occupancy_from_simple_18_counts_pieces():
    board = _fen_to_tensor(INITIAL_FEN)
    occ = occupancy_from_simple_18(board)
    assert occ.shape == (1, 8, 8)
    # Initial position has 32 pieces.
    assert float(occ.sum().item()) == 32.0


def test_num_directions_consistent():
    assert NUM_DIRECTIONS == 8
    assert len(RAY_DIRECTIONS) == 8


def test_ray_pool_terminates_at_blocker():
    # Construct a 2D board with a "value" on the source and a blocker on
    # the adjacent square along the south direction.
    features = torch.zeros(1, 1, 8, 8)
    features[0, 0, 0, 0] = 1.0  # row 0, col 0 has a feature value
    occupancy = torch.zeros(1, 8, 8)
    occupancy[0, 1, 0] = 1.0  # blocker directly south of source
    gamma = torch.tensor([1.0] * 8)  # no decay
    pooled = ray_pool(features, occupancy, gamma, max_ray_length=7, use_occlusion=True)
    # Square (2, 0) is two steps south of source. With a blocker at (1, 0),
    # the unblocked coefficient becomes 0 after step 1, so (2, 0) should
    # have no contribution from the source via the south ray.
    # Direction 0 is (-1, 0) (N); direction 4 is (1, 0) (S). For the cell
    # at (2, 0) looking south for its source, the contribution we want to
    # check is from direction (-1, 0) (N) applied to that cell.
    # (2, 0) looks N to (1, 0) (blocker — but the source feature is at (0, 0)).
    # With occlusion enabled, the contribution of (0, 0) reaches (2, 0)
    # only if (1, 0) is unblocked. Since (1, 0) is blocked, (2, 0) sees
    # only (1, 0) (which has zero feature value).
    cell_2_0_north = pooled[0, 0, 0, 2, 0]
    assert float(cell_2_0_north.item()) == pytest.approx(0.0, abs=1.0e-6)


def test_ray_pool_propagates_when_unblocked():
    features = torch.zeros(1, 1, 8, 8)
    features[0, 0, 0, 0] = 1.0  # value at (0, 0)
    occupancy = torch.zeros(1, 8, 8)  # empty
    gamma = torch.tensor([1.0] * 8)
    pooled = ray_pool(features, occupancy, gamma, max_ray_length=7, use_occlusion=True)
    # Cell (2, 0) looking N should see the feature at (0, 0) since the
    # ray is unblocked.
    cell_2_0_north = pooled[0, 0, 0, 2, 0]
    assert float(cell_2_0_north.item()) == pytest.approx(1.0, abs=1.0e-6)


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
        "raypool_active_squares",
        "raypool_ray_energy",
        "raypool_max_dir_energy",
        "raypool_gamma_mean",
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


def test_drop_occlusion_changes_ray_energy():
    # Construct a position with a known long ray (white rook on d1, otherwise empty).
    fen = "4k3/8/8/8/8/8/8/3R3K w - - 0 1"
    board = torch.from_numpy(fen_to_simple_18(fen)).unsqueeze(0)
    cfg = _small_config()
    torch.manual_seed(0)
    model_normal = build_model(REGISTRY_KEY, cfg).eval()
    torch.manual_seed(0)
    cfg_drop = dict(cfg)
    cfg_drop["ablation"] = "drop_occlusion"
    model_drop = build_model(REGISTRY_KEY, cfg_drop).eval()
    with torch.no_grad():
        out_normal = model_normal(board)
        out_drop = model_drop(board)
    # The energy should differ between the two modes for a position with
    # a sliding piece (since the blocker mask changes the prefix sum).
    assert not torch.isclose(
        out_normal["raypool_ray_energy"], out_drop["raypool_ray_energy"], atol=1.0e-6
    )


def test_zero_rays_zeros_primitive_delta():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_rays")).eval()
    boards = _fen_to_tensor(BACK_RANK_MATE_FEN)
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


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
        "drop_occlusion",
        "shuffle_directions",
        "zero_rays",
    }
    assert set(ALLOWED_ABLATIONS) == expected


def test_config_yaml_loads_correctly():
    config_path = IDEA_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["idea_id"] == "p026"
    assert config["mode"] == "puzzle_binary"
    assert config["device"] == "nvidia"
    assert config["model"]["name"] == REGISTRY_KEY


def test_rejects_non_simple_18_input():
    with pytest.raises(ValueError):
        RayCastObstaclePoolHead(input_channels=12)


def test_rejects_invalid_max_ray_length():
    with pytest.raises(ValueError):
        RayCastObstaclePoolHead(max_ray_length=0)
    with pytest.raises(ValueError):
        RayCastObstaclePoolHead(max_ray_length=8)


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
    model = build_ray_cast_obstacle_pool_head_from_config(cfg)
    assert isinstance(model, RayCastObstaclePoolHead)
    assert model.trunk.channels == 24
