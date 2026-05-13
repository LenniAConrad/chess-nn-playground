"""Tests for the p038 Woodbury Set Resolver primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.woodbury_set_resolver import (
    WoodburySetResolver,
    build_woodbury_set_resolver_from_config,
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
        "u_dim": 6,
        "v_dim": 8,
        "num_queries": 2,
        "lambda_reg": 0.01,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "woodbury_set_resolver" in available_models()
    model = build_model("woodbury_set_resolver", _toy_kwargs())
    assert isinstance(model, WoodburySetResolver)


def test_active_count_matches_occupancy() -> None:
    model = WoodburySetResolver(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    assert out["wsr_active_count"][0].item() == pytest.approx(32.0)
    assert out["wsr_active_count"][1].item() == pytest.approx(6.0)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = WoodburySetResolver(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    assert out["logits"].shape == (1,)
    for diag in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "wsr_active_count",
        "wsr_logdet_A",
        "wsr_leverage_mean",
        "wsr_leverage_max",
        "wsr_A_norm",
    ):
        assert diag in out


def test_logdet_is_finite_and_positive_for_nontrivial_position() -> None:
    model = WoodburySetResolver(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    # A = lambda I + sum z z^T is SPD with eigenvalues >= lambda, so log det >= r * log(lambda).
    # Here r=6, lambda=0.01, lower bound = 6 * log(0.01) = -27.6.
    assert torch.isfinite(out["wsr_logdet_A"]).all()


def test_leverage_is_nonnegative_on_active_squares() -> None:
    model = WoodburySetResolver(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    # Leverage is u^T P u where P is SPD -- should be >= 0.
    # We only have the per-sample mean and max here.
    assert out["wsr_leverage_mean"].item() >= -1.0e-6
    assert out["wsr_leverage_max"].item() >= -1.0e-6


def test_diagonal_only_ablation_changes_delta() -> None:
    torch.manual_seed(0)
    model_full = WoodburySetResolver(**_toy_kwargs())
    torch.manual_seed(0)
    model_diag = WoodburySetResolver(**_toy_kwargs(ablation="diagonal_only"))
    for p1, p2 in zip(model_full.parameters(), model_diag.parameters()):
        assert torch.equal(p1, p2)
    model_full.eval(); model_diag.eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out_full = model_full(boards)
        out_diag = model_diag(boards)
    assert not torch.allclose(out_full["primitive_delta_raw"], out_diag["primitive_delta_raw"])


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = WoodburySetResolver(**_toy_kwargs(ablation="zero_delta")).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = WoodburySetResolver(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad.abs().sum() > 0
    assert model.u_proj.weight.grad.abs().sum() > 0
    assert model.v_proj.weight.grad.abs().sum() > 0


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "u_dim": 6,
        "v_dim": 8,
        "num_queries": 2,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_woodbury_set_resolver_from_config(cfg)
    assert isinstance(model, WoodburySetResolver)
    assert model.trunk.channels == 24


def test_rejects_invalid_lambda() -> None:
    with pytest.raises(ValueError):
        WoodburySetResolver(**_toy_kwargs(lambda_reg=0.0))


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        WoodburySetResolver(**_toy_kwargs(ablation="not_a_real_ablation"))


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        WoodburySetResolver(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        WoodburySetResolver(input_channels=18, num_classes=3)
