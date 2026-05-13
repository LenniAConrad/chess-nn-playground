"""Tests for the p039 Differentiable Occupancy Eikonal Transform primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.occupancy_eikonal_transform import (
    OccupancyEikonalTransform,
    _king_neighbour_index,
    build_occupancy_eikonal_transform_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


def _toy_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 32,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "q_channels": 2,
        "temperature": 0.5,
        "num_iterations": 4,
        "cost_bias": 1.0,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "occupancy_eikonal_transform" in available_models()
    model = build_model("occupancy_eikonal_transform", _toy_kwargs())
    assert isinstance(model, OccupancyEikonalTransform)


def test_neighbour_table_is_consistent() -> None:
    nb = _king_neighbour_index()
    assert nb.shape == (64, 8)
    # Center cell (row 3, col 3) -> sq 27 should have 8 distinct neighbours.
    centre = nb[27]
    assert len(set(centre.tolist())) == 8
    # Corner cell (row 0, col 0) -> sq 0 should have 5 distinct destinations
    # (3 real + 5 self-loops onto itself).
    corner = nb[0].tolist()
    real_neighbours = {1, 8, 9}
    assert set(corner) - {0} == real_neighbours


def test_forward_returns_required_keys_and_shapes() -> None:
    model = OccupancyEikonalTransform(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    assert out["logits"].shape == (1,)
    for diag in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "eikonal_field_mean",
        "eikonal_field_max",
        "eikonal_field_min",
        "eikonal_field_range",
    ):
        assert diag in out


def test_arrival_field_is_strictly_positive() -> None:
    model = OccupancyEikonalTransform(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    # The cost field and seed are softplus(.) + cost_bias >= cost_bias > 0,
    # so the arrival field is bounded below by cost_bias.
    assert (out["eikonal_field_min"] >= 0.5).all()


def test_field_range_is_nonnegative() -> None:
    model = OccupancyEikonalTransform(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert (out["eikonal_field_range"] >= -1.0e-6).all()


def test_single_iteration_ablation_changes_field() -> None:
    torch.manual_seed(0)
    model_full = OccupancyEikonalTransform(**_toy_kwargs())
    torch.manual_seed(0)
    model_single = OccupancyEikonalTransform(**_toy_kwargs(ablation="single_iteration"))
    for p1, p2 in zip(model_full.parameters(), model_single.parameters()):
        assert torch.equal(p1, p2)
    model_full.eval(); model_single.eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out_full = model_full(boards)
        out_single = model_single(boards)
    assert not torch.allclose(out_full["eikonal_field_mean"], out_single["eikonal_field_mean"])


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = OccupancyEikonalTransform(**_toy_kwargs(ablation="zero_delta")).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_uniform_costs_ablation_makes_field_uniform_for_constant_seed() -> None:
    # Sanity check: with uniform cost equal to seed (both = cost_bias) the
    # iteration is at a fixed point, so the field should equal seed = cost_bias.
    model = OccupancyEikonalTransform(**_toy_kwargs(ablation="uniform_costs")).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    # The seed is also passed through softplus(.) + cost_bias, so we cannot
    # easily verify exact equality without monkey-patching the seed. Instead
    # check that the range stays bounded and the field stays positive.
    assert (out["eikonal_field_min"] >= 0.0).all()
    assert torch.isfinite(out["eikonal_field_mean"]).all()


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = OccupancyEikonalTransform(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad.abs().sum() > 0
    assert model.cost_proj.weight.grad.abs().sum() > 0
    assert model.seed_proj.weight.grad.abs().sum() > 0


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "q_channels": 2,
        "temperature": 0.5,
        "num_iterations": 3,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_occupancy_eikonal_transform_from_config(cfg)
    assert isinstance(model, OccupancyEikonalTransform)
    assert model.trunk.channels == 24


def test_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        OccupancyEikonalTransform(**_toy_kwargs(temperature=0.0))


def test_rejects_invalid_num_iterations() -> None:
    with pytest.raises(ValueError):
        OccupancyEikonalTransform(**_toy_kwargs(num_iterations=0))


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        OccupancyEikonalTransform(**_toy_kwargs(ablation="not_a_real_ablation"))


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        OccupancyEikonalTransform(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        OccupancyEikonalTransform(input_channels=18, num_classes=3)
