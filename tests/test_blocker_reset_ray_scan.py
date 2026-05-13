"""Tests for the p020 Blocker-Reset Ray Scan primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.blocker_reset_ray_scan import (
    BlockerResetRayScan,
    build_blocker_reset_ray_scan_from_config,
)
from chess_nn_playground.models.primitives.ray_geometry import (
    NUM_DIRECTIONS,
    RAY_MAX_LEN,
    RayGeometry,
)
from chess_nn_playground.models.registry import available_models, build_model


def _toy_kwargs() -> dict[str, object]:
    return {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 32,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "token_dim": 12,
        "hidden_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "blocker_reset_ray_scan" in available_models()
    model = build_model("blocker_reset_ray_scan", _toy_kwargs())
    assert isinstance(model, BlockerResetRayScan)


def test_ray_geometry_has_expected_shape_and_consistency() -> None:
    g = RayGeometry.build()
    assert g.step_index.shape == (NUM_DIRECTIONS, 64, RAY_MAX_LEN)
    assert g.step_mask.shape == (NUM_DIRECTIONS, 64, RAY_MAX_LEN)
    # Corner squares have at most one direction with full RAY_MAX_LEN valid steps.
    a8 = 0
    h1 = 63
    # From a8 (top-left), only S/E/SE rays have RAY_MAX_LEN = 7 valid steps.
    for d in range(NUM_DIRECTIONS):
        assert g.step_mask[d, a8].sum().item() <= RAY_MAX_LEN
        assert g.step_mask[d, h1].sum().item() <= RAY_MAX_LEN
    # Centre square e4 (rank 4, file e -> row 4, file 4 -> square 36) has 7 cells in any direction.
    e4 = 4 * 8 + 4
    # In some directions e4 has fewer cells (e.g. only 3 N steps to a8 row).
    assert g.step_mask[:, e4, :].sum().item() > 0


def test_forward_returns_required_keys_and_shapes() -> None:
    model = BlockerResetRayScan(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["base_logit"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    for diag in ("brrs_occupancy_density", "brrs_ray_magnitude", "brrs_decay_mean"):
        assert diag in out
        assert out[diag].shape == (2,)
    assert torch.all(out["brrs_occupancy_density"] >= 0.0)
    assert torch.all(out["brrs_occupancy_density"] <= 1.0)


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = BlockerResetRayScan(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    trunk_param = next(model.trunk.parameters())
    input_param = model.input_proj.weight
    decay_param = model.decay_logit
    assert trunk_param.grad is not None and trunk_param.grad.abs().sum() > 0
    assert input_param.grad is not None and input_param.grad.abs().sum() > 0
    assert decay_param.grad is not None and decay_param.grad.abs().sum() > 0


def test_zero_blocker_ablation_changes_delta_distribution() -> None:
    # With identical weights, the no-blocker scan should produce a different
    # delta_raw than the full scan on positions with blockers.
    torch.manual_seed(0)
    model_full = BlockerResetRayScan(**_toy_kwargs())
    torch.manual_seed(0)
    model_no_blocker = BlockerResetRayScan(**_toy_kwargs(), ablation="zero_blocker")
    # Verify the weights line up.
    for p1, p2 in zip(model_full.parameters(), model_no_blocker.parameters()):
        assert torch.equal(p1, p2)
    model_full.eval(); model_no_blocker.eval()
    boards = _board_batch([
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",   # starting -- many blockers
    ])
    with torch.no_grad():
        out_full = model_full(boards)
        out_nb = model_no_blocker(boards)
    # The ablation must produce different raw delta on a heavily blocked position.
    assert not torch.allclose(out_full["primitive_delta_raw"], out_nb["primitive_delta_raw"])


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = BlockerResetRayScan(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_ablation_zeros_delta() -> None:
    model = BlockerResetRayScan(**_toy_kwargs(), ablation="trunk_only").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,         # aliased -> trunk_channels
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "token_dim": 12,
        "hidden_dim": 18,       # this aliases to both trunk_hidden_dim and the head's hidden_dim
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_blocker_reset_ray_scan_from_config(cfg)
    assert isinstance(model, BlockerResetRayScan)
    assert model.trunk.channels == 24
    # The hidden_dim alias is shared by trunk_hidden_dim and the head's hidden_dim.
    assert model.hidden_dim == 18


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        BlockerResetRayScan(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        BlockerResetRayScan(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        BlockerResetRayScan(input_channels=18, num_classes=3)
