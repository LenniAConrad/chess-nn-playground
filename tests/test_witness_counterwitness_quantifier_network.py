"""Tests for the p005 Witness-Counterwitness Quantifier Network."""

from __future__ import annotations

import pytest
import torch

from chess_nn_playground.models.primitives.witness_counterwitness_quantifier_network import (
    ALLOWED_ABLATIONS,
    WitnessCounterwitnessQuantifierNetwork,
    build_witness_counterwitness_quantifier_network_from_config,
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
        "compat_dim": 8,
    }


def _toy_board() -> torch.Tensor:
    board = torch.zeros(2, 18, 8, 8)
    board[:, 0, 6, :] = 1.0
    board[:, 5, 7, 4] = 1.0
    board[:, 11, 0, 4] = 1.0
    board[:, 12] = 1.0
    return board


def test_model_is_registered() -> None:
    assert "witness_counterwitness_quantifier_network" in available_models()
    model = build_model("witness_counterwitness_quantifier_network", _toy_kwargs())
    assert isinstance(model, WitnessCounterwitnessQuantifierNetwork)


def test_forward_returns_expected_keys() -> None:
    model = WitnessCounterwitnessQuantifierNetwork(**_toy_kwargs()).eval()
    out = model(_toy_board())
    for key in (
        "wcq_value",
        "wcq_max_margin",
        "wcq_min_margin",
        "wcq_counter_envelope_max",
        "wcq_witness_entropy",
        "wcq_best_witness_index",
        "wcq_best_counter_index",
        "wcq_claim_max",
    ):
        assert key in out
        assert out[key].shape == (2,)


def test_backward_gradients_flow() -> None:
    torch.manual_seed(0)
    model = WitnessCounterwitnessQuantifierNetwork(**_toy_kwargs())
    out = model(_toy_board())
    out["logits"].pow(2).mean().backward()
    for param in (
        next(model.trunk.parameters()),
        next(model.delta_mlp.parameters()),
        next(model.candidate_pool.parameters()),
        next(model.reply_pool.parameters()),
        next(model.claim_head.parameters()),
        next(model.counter_head.parameters()),
    ):
        assert param.grad is not None
        assert param.grad.abs().sum() > 0


def test_zero_delta_recovers_trunk_logit() -> None:
    model = WitnessCounterwitnessQuantifierNetwork(**_toy_kwargs(), ablation="zero_delta").eval()
    out = model(_toy_board())
    assert torch.allclose(out["logits"], out["base_logit"])


def test_no_counter_branch_zero_counter_constant_offset() -> None:
    """With zeroed counter scores the counter envelope is a constant offset.

    The soft forall over R zeros evaluates to ``tau_forall * log(R)`` (not 0),
    so ``wcq_max_margin`` is offset from ``wcq_claim_max`` by exactly that
    constant. The candidate ordering is preserved.
    """
    import math
    kwargs = _toy_kwargs()
    model = WitnessCounterwitnessQuantifierNetwork(**kwargs, ablation="no_counter_branch").eval()
    out = model(_toy_board())
    assert torch.isfinite(out["wcq_value"]).all()
    expected_offset = float(kwargs["tau_forall"]) * math.log(float(kwargs["num_replies"])) \
        if "tau_forall" in kwargs else 0.20 * math.log(4)
    diff = (out["wcq_max_margin"] - (out["wcq_claim_max"] - expected_offset)).abs()
    assert diff.max().item() < 1.0e-3


def test_max_claim_only_uses_only_claim() -> None:
    model = WitnessCounterwitnessQuantifierNetwork(**_toy_kwargs(), ablation="max_claim_only").eval()
    out = model(_toy_board())
    assert torch.allclose(out["wcq_value"], out["wcq_claim_max"], atol=1.0e-4)


def test_all_documented_ablations_are_allowed() -> None:
    for ab in ALLOWED_ABLATIONS:
        model = WitnessCounterwitnessQuantifierNetwork(**_toy_kwargs(), ablation=ab).eval()
        out = model(_toy_board())
        assert torch.isfinite(out["logits"]).all()


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        WitnessCounterwitnessQuantifierNetwork(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        WitnessCounterwitnessQuantifierNetwork(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        WitnessCounterwitnessQuantifierNetwork(input_channels=18, num_classes=3)


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
    model = build_witness_counterwitness_quantifier_network_from_config(cfg)
    assert isinstance(model, WitnessCounterwitnessQuantifierNetwork)
    assert model.trunk.channels == 20
