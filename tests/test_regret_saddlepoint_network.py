"""Tests for the p002 Regret Saddlepoint Network."""

from __future__ import annotations

import pytest
import torch

from chess_nn_playground.models.primitives.regret_saddlepoint_network import (
    ALLOWED_ABLATIONS,
    RegretSaddlepointNetwork,
    build_regret_saddlepoint_network_from_config,
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
        "num_candidates": 5,
        "num_replies": 4,
        "token_dim": 12,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "solver_iters": 8,
    }


def _toy_board() -> torch.Tensor:
    board = torch.zeros(2, 18, 8, 8)
    board[:, 0, 6, :] = 1.0
    board[:, 5, 7, 4] = 1.0
    board[:, 11, 0, 4] = 1.0
    board[:, 12] = 1.0
    return board


def test_model_is_registered() -> None:
    assert "regret_saddlepoint_network" in available_models()
    model = build_model("regret_saddlepoint_network", _toy_kwargs())
    assert isinstance(model, RegretSaddlepointNetwork)


def test_forward_returns_expected_keys() -> None:
    model = RegretSaddlepointNetwork(**_toy_kwargs()).eval()
    out = model(_toy_board())
    for key in (
        "rsp_saddle_value",
        "rsp_attacker_regret",
        "rsp_defender_regret",
        "rsp_exploitability",
        "rsp_attacker_entropy",
        "rsp_defender_entropy",
        "rsp_best_witness_index",
        "rsp_best_reply_index",
    ):
        assert key in out
        assert out[key].shape == (2,)
    assert torch.all(out["rsp_exploitability"] >= -1.0e-3)


def test_backward_gradients_flow() -> None:
    torch.manual_seed(0)
    model = RegretSaddlepointNetwork(**_toy_kwargs())
    out = model(_toy_board())
    out["logits"].pow(2).mean().backward()
    for param in (
        next(model.trunk.parameters()),
        next(model.delta_mlp.parameters()),
        next(model.candidate_pool.parameters()),
        next(model.reply_pool.parameters()),
    ):
        assert param.grad is not None
        assert param.grad.abs().sum() > 0


def test_zero_delta_recovers_trunk_logit() -> None:
    model = RegretSaddlepointNetwork(**_toy_kwargs(), ablation="zero_delta").eval()
    out = model(_toy_board())
    assert torch.allclose(out["logits"], out["base_logit"])


def test_pure_max_min_returns_finite_value() -> None:
    model = RegretSaddlepointNetwork(**_toy_kwargs(), ablation="pure_max_min").eval()
    out = model(_toy_board())
    assert torch.isfinite(out["rsp_saddle_value"]).all()


def test_uniform_payoff_zeros_exploitability() -> None:
    model = RegretSaddlepointNetwork(**_toy_kwargs(), ablation="uniform_payoff").eval()
    out = model(_toy_board())
    assert torch.all(out["rsp_exploitability"] < 1.0e-3)


def test_all_documented_ablations_are_allowed() -> None:
    for ab in ALLOWED_ABLATIONS:
        model = RegretSaddlepointNetwork(**_toy_kwargs(), ablation=ab).eval()
        out = model(_toy_board())
        assert torch.isfinite(out["logits"]).all()


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        RegretSaddlepointNetwork(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        RegretSaddlepointNetwork(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        RegretSaddlepointNetwork(input_channels=18, num_classes=3)


def test_builder_accepts_aliases() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 20,
        "hidden_dim": 40,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "num_candidates": 5,
        "num_replies": 4,
        "token_dim": 12,
        "head_hidden_dim": 16,
    }
    model = build_regret_saddlepoint_network_from_config(cfg)
    assert isinstance(model, RegretSaddlepointNetwork)
    assert model.trunk.channels == 20
