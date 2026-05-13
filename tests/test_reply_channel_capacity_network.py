"""Tests for the p003 Reply Channel Capacity Network."""

from __future__ import annotations

import pytest
import torch

from chess_nn_playground.models.primitives.reply_channel_capacity_network import (
    ALLOWED_ABLATIONS,
    ReplyChannelCapacityNetwork,
    build_reply_channel_capacity_network_from_config,
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
    assert "reply_channel_capacity_network" in available_models()
    model = build_model("reply_channel_capacity_network", _toy_kwargs())
    assert isinstance(model, ReplyChannelCapacityNetwork)


def test_forward_returns_expected_keys() -> None:
    model = ReplyChannelCapacityNetwork(**_toy_kwargs()).eval()
    out = model(_toy_board())
    for key in (
        "rcc_capacity_nats",
        "rcc_capacity_bits",
        "rcc_conditional_entropy",
        "rcc_output_entropy",
        "rcc_capacity_gap",
        "rcc_prior_entropy",
        "rcc_marginal_entropy",
    ):
        assert key in out
        assert out[key].shape == (2,)
    # capacity is non-negative
    assert torch.all(out["rcc_capacity_nats"] >= -1.0e-4)
    assert torch.all(out["rcc_capacity_gap"] >= -1.0e-4)


def test_backward_gradients_flow() -> None:
    torch.manual_seed(0)
    model = ReplyChannelCapacityNetwork(**_toy_kwargs())
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
    model = ReplyChannelCapacityNetwork(**_toy_kwargs(), ablation="zero_delta").eval()
    out = model(_toy_board())
    assert torch.allclose(out["logits"], out["base_logit"])


def test_duplicate_rows_zero_capacity() -> None:
    model = ReplyChannelCapacityNetwork(**_toy_kwargs(), ablation="duplicate_rows").eval()
    out = model(_toy_board())
    assert torch.all(out["rcc_capacity_nats"] < 1.0e-2)


def test_uniform_replies_zero_capacity() -> None:
    model = ReplyChannelCapacityNetwork(**_toy_kwargs(), ablation="uniform_replies").eval()
    out = model(_toy_board())
    assert torch.all(out["rcc_capacity_nats"] < 1.0e-2)


def test_all_documented_ablations_are_allowed() -> None:
    for ab in ALLOWED_ABLATIONS:
        model = ReplyChannelCapacityNetwork(**_toy_kwargs(), ablation=ab).eval()
        out = model(_toy_board())
        assert torch.isfinite(out["logits"]).all()


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        ReplyChannelCapacityNetwork(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        ReplyChannelCapacityNetwork(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        ReplyChannelCapacityNetwork(input_channels=18, num_classes=3)


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
    model = build_reply_channel_capacity_network_from_config(cfg)
    assert isinstance(model, ReplyChannelCapacityNetwork)
    assert model.trunk.channels == 20
