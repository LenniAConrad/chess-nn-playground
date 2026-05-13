"""Tests for the p041 Truncated Exterior Product Pool primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.truncated_exterior_product_pool import (
    TruncatedExteriorProductPool,
    _grade_indices,
    build_truncated_exterior_product_pool_from_config,
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
        "r": 3,
        "max_grade": 2,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "truncated_exterior_product_pool" in available_models()
    model = build_model("truncated_exterior_product_pool", _toy_kwargs())
    assert isinstance(model, TruncatedExteriorProductPool)


def test_grade_indices_match_binomial_counts() -> None:
    # Grade-k dimensions equal C(r, k).
    assert len(_grade_indices(3, 0)) == 1
    assert len(_grade_indices(3, 1)) == 3
    assert len(_grade_indices(3, 2)) == 3
    assert len(_grade_indices(3, 3)) == 1
    assert len(_grade_indices(4, 2)) == 6


def test_wedge_grade_1_equals_sum_for_two_tokens() -> None:
    # M^{(1)} should equal sum_i a_i z_i over active tokens (the
    # "first-order" part of the wedge accumulator).
    torch.manual_seed(0)
    model = TruncatedExteriorProductPool(**_toy_kwargs()).eval()
    # Direct test of the _exterior_pool helper.
    B, n, r = 1, 3, 3
    z = torch.randn(B, n, r) * 0.3
    mask = torch.ones(B, n)
    grades = model._exterior_pool(z, mask)
    expected_g1 = (z * mask.unsqueeze(-1)).sum(dim=1)  # (B, r)
    assert torch.allclose(grades[1], expected_g1, atol=1.0e-5)


def test_wedge_grade_2_is_antisymmetric_on_two_tokens() -> None:
    # For two tokens z_a, z_b the grade-2 component should equal the
    # exterior product z_a ^ z_b. We check that swapping z_a and z_b
    # flips the sign of every component.
    torch.manual_seed(0)
    model = TruncatedExteriorProductPool(**_toy_kwargs(r=3, max_grade=2)).eval()
    B, r = 1, 3
    z_a = torch.tensor([[[0.5, 0.2, -0.1]]])  # (1, 1, 3)
    z_b = torch.tensor([[[-0.3, 0.4, 0.1]]])
    z_ab = torch.cat([z_a, z_b], dim=1)
    z_ba = torch.cat([z_b, z_a], dim=1)
    mask = torch.ones(B, 2)
    g_ab = model._exterior_pool(z_ab, mask)[2]  # (B, 3)
    g_ba = model._exterior_pool(z_ba, mask)[2]
    # Wedge product is antisymmetric so the grade-2 part flips sign.
    assert torch.allclose(g_ab, -g_ba, atol=1.0e-6)


def test_wedge_grade_2_vanishes_on_colinear_tokens() -> None:
    # For z_a = lambda * z_b the grade-2 wedge is zero. We construct two
    # parallel tokens and confirm M^{(2)} ~= 0.
    model = TruncatedExteriorProductPool(**_toy_kwargs(r=3, max_grade=2)).eval()
    B, r = 1, 3
    base = torch.tensor([[0.4, -0.1, 0.2]])  # (1, 3)
    z_a = base.unsqueeze(0)  # (1, 1, 3)
    z_b = (2.0 * base).unsqueeze(0)
    z = torch.cat([z_a, z_b], dim=1)
    mask = torch.ones(B, 2)
    grades = model._exterior_pool(z, mask)
    assert torch.allclose(grades[2], torch.zeros_like(grades[2]), atol=1.0e-6)


def test_active_count_matches_occupancy() -> None:
    model = TruncatedExteriorProductPool(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    assert out["tepp_active_count"][0].item() == pytest.approx(32.0)
    assert out["tepp_active_count"][1].item() == pytest.approx(6.0)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = TruncatedExteriorProductPool(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    assert out["logits"].shape == (1,)
    for diag in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "tepp_active_count",
        "tepp_grade_max_magnitude",
        "tepp_grade_mean_magnitude",
        "tepp_grade_0_magnitude",
        "tepp_grade_1_magnitude",
        "tepp_grade_2_magnitude",
    ):
        assert diag in out


def test_first_order_only_ablation_changes_delta() -> None:
    torch.manual_seed(0)
    model_full = TruncatedExteriorProductPool(**_toy_kwargs())
    torch.manual_seed(0)
    model_first = TruncatedExteriorProductPool(**_toy_kwargs(ablation="first_order_only"))
    for p1, p2 in zip(model_full.parameters(), model_first.parameters()):
        assert torch.equal(p1, p2)
    model_full.eval(); model_first.eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out_full = model_full(boards)
        out_first = model_first(boards)
    assert not torch.allclose(out_full["primitive_delta_raw"], out_first["primitive_delta_raw"])


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = TruncatedExteriorProductPool(**_toy_kwargs(ablation="zero_delta")).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = TruncatedExteriorProductPool(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad.abs().sum() > 0
    assert model.z_proj.weight.grad.abs().sum() > 0


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "r": 3,
        "max_grade": 2,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_truncated_exterior_product_pool_from_config(cfg)
    assert isinstance(model, TruncatedExteriorProductPool)
    assert model.trunk.channels == 24


def test_rejects_invalid_r() -> None:
    with pytest.raises(ValueError):
        TruncatedExteriorProductPool(**_toy_kwargs(r=0))
    with pytest.raises(ValueError):
        TruncatedExteriorProductPool(**_toy_kwargs(r=9))


def test_rejects_invalid_max_grade() -> None:
    with pytest.raises(ValueError):
        TruncatedExteriorProductPool(**_toy_kwargs(r=3, max_grade=4))
    with pytest.raises(ValueError):
        TruncatedExteriorProductPool(**_toy_kwargs(r=3, max_grade=0))


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        TruncatedExteriorProductPool(**_toy_kwargs(ablation="not_a_real_ablation"))


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        TruncatedExteriorProductPool(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        TruncatedExteriorProductPool(input_channels=18, num_classes=3)
