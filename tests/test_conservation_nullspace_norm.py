"""Tests for the p040 Conservation-Nullspace Normalization primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.conservation_nullspace_norm import (
    ConservationNullspaceNorm,
    _build_charge_matrix,
    build_conservation_nullspace_norm_from_config,
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
        "latent_dim": 8,
        "epsilon": 0.001,
        "weight_bias": 1.0,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "conservation_nullspace_norm" in available_models()
    model = build_model("conservation_nullspace_norm", _toy_kwargs())
    assert isinstance(model, ConservationNullspaceNorm)


def test_charge_matrix_shape_and_constant_column() -> None:
    C = _build_charge_matrix()
    assert C.shape == (64, 8)
    # Constant column is all ones.
    assert torch.allclose(C[:, 0], torch.ones(64))
    # File column at the edges is +/- 1.
    assert C[0, 1].item() == pytest.approx(-1.0)
    assert C[7, 1].item() == pytest.approx(1.0)
    # Corner indicator: 4 corners set.
    assert C[:, 7].sum().item() == pytest.approx(4.0)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = ConservationNullspaceNorm(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    assert out["logits"].shape == (1,)
    for diag in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "cnnorm_residual_norm",
        "cnnorm_x_norm",
        "cnnorm_explained_frac",
        "cnnorm_sum_weight",
        "cnnorm_sigma_mean",
    ):
        assert diag in out


def test_explained_fraction_is_in_unit_interval() -> None:
    model = ConservationNullspaceNorm(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert (out["cnnorm_explained_frac"] >= 0.0).all()
    assert (out["cnnorm_explained_frac"] <= 1.0).all()


def test_no_projection_ablation_makes_explained_frac_zero() -> None:
    model = ConservationNullspaceNorm(**_toy_kwargs(ablation="no_projection")).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    # In the no_projection mode the residual is the full latent, so the
    # "explained fraction" collapses to ~eps / (x_norm + eps) ~ 0.
    assert (out["cnnorm_explained_frac"].abs() < 1.0e-2).all()


def test_uniform_weights_ablation_changes_residual() -> None:
    torch.manual_seed(0)
    model_full = ConservationNullspaceNorm(**_toy_kwargs())
    torch.manual_seed(0)
    model_uniform = ConservationNullspaceNorm(**_toy_kwargs(ablation="uniform_weights"))
    for p1, p2 in zip(model_full.parameters(), model_uniform.parameters()):
        assert torch.equal(p1, p2)
    model_full.eval(); model_uniform.eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out_full = model_full(boards)
        out_uniform = model_uniform(boards)
    assert not torch.allclose(out_full["cnnorm_residual_norm"], out_uniform["cnnorm_residual_norm"])


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = ConservationNullspaceNorm(**_toy_kwargs(ablation="zero_delta")).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = ConservationNullspaceNorm(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad.abs().sum() > 0
    assert model.latent_proj.weight.grad.abs().sum() > 0
    assert model.weight_proj.weight.grad.abs().sum() > 0


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "latent_dim": 8,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_conservation_nullspace_norm_from_config(cfg)
    assert isinstance(model, ConservationNullspaceNorm)
    assert model.trunk.channels == 24


def test_rejects_invalid_epsilon() -> None:
    with pytest.raises(ValueError):
        ConservationNullspaceNorm(**_toy_kwargs(epsilon=0.0))


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        ConservationNullspaceNorm(**_toy_kwargs(ablation="not_a_real_ablation"))


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        ConservationNullspaceNorm(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        ConservationNullspaceNorm(input_channels=18, num_classes=3)
