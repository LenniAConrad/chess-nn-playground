"""Tests for the p001 Pareto Antichain Frontier Network."""

from __future__ import annotations

import pytest
import torch

from chess_nn_playground.models.primitives.pareto_antichain_frontier_network import (
    ALLOWED_ABLATIONS,
    ParetoAntichainFrontierNetwork,
    build_pareto_antichain_frontier_network_from_config,
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
        "num_candidates": 6,
        "token_dim": 12,
        "utility_channels": 4,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }


def _toy_board() -> torch.Tensor:
    board = torch.zeros(2, 18, 8, 8)
    board[:, 0, 6, :] = 1.0
    board[:, 5, 7, 4] = 1.0
    board[:, 11, 0, 4] = 1.0
    board[:, 12] = 1.0
    return board


def test_model_is_registered_with_expected_key() -> None:
    assert "pareto_antichain_frontier_network" in available_models()
    model = build_model("pareto_antichain_frontier_network", _toy_kwargs())
    assert isinstance(model, ParetoAntichainFrontierNetwork)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = ParetoAntichainFrontierNetwork(**_toy_kwargs()).eval()
    board = _toy_board()
    out = model(board)
    assert out["logits"].shape == (2,)
    assert out["base_logit"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    assert out["primitive_gate"].shape == (2,)
    assert torch.all(out["primitive_gate"] >= 0.0)
    assert torch.all(out["primitive_gate"] <= 1.0)
    for key in (
        "pafr_frontier_width",
        "pafr_frontier_entropy",
        "pafr_max_nondominated_prob",
        "pafr_summary_norm",
        "pafr_utility_mean",
        "pafr_utility_max",
    ):
        assert key in out, f"missing diagnostic key {key}"
        assert out[key].shape == (2,)


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    torch.manual_seed(0)
    model = ParetoAntichainFrontierNetwork(**_toy_kwargs())
    board = _toy_board()
    out = model(board)
    out["logits"].pow(2).mean().backward()
    trunk_param = next(model.trunk.parameters())
    head_param = next(model.delta_mlp.parameters())
    gate_param = next(model.gate_mlp.parameters())
    pool_param = next(model.candidate_pool.parameters())
    assert trunk_param.grad is not None and trunk_param.grad.abs().sum() > 0
    assert head_param.grad is not None and head_param.grad.abs().sum() > 0
    assert gate_param.grad is not None and gate_param.grad.abs().sum() > 0
    assert pool_param.grad is not None and pool_param.grad.abs().sum() > 0


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = ParetoAntichainFrontierNetwork(**_toy_kwargs(), ablation="zero_delta").eval()
    out = model(_toy_board())
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_ablation_zero_delta() -> None:
    model = ParetoAntichainFrontierNetwork(**_toy_kwargs(), ablation="trunk_only").eval()
    out = model(_toy_board())
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


def test_disable_gate_holds_gate_at_one() -> None:
    model = ParetoAntichainFrontierNetwork(**_toy_kwargs(), ablation="disable_gate").eval()
    out = model(_toy_board())
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_all_documented_ablations_are_allowed() -> None:
    for ab in ALLOWED_ABLATIONS:
        model = ParetoAntichainFrontierNetwork(**_toy_kwargs(), ablation=ab).eval()
        out = model(_toy_board())
        assert torch.isfinite(out["logits"]).all()


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        ParetoAntichainFrontierNetwork(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        ParetoAntichainFrontierNetwork(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        ParetoAntichainFrontierNetwork(input_channels=18, num_classes=3)


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "num_candidates": 6,
        "token_dim": 12,
        "utility_channels": 4,
        "head_hidden_dim": 16,
    }
    model = build_pareto_antichain_frontier_network_from_config(cfg)
    assert isinstance(model, ParetoAntichainFrontierNetwork)
    assert model.trunk.channels == 24
