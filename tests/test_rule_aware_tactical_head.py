"""Tests for the i248 Rule-Aware Tactical Head model."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.data.terminal_state import TSDP_FEATURE_NAMES
from chess_nn_playground.models.primitives.rule_aware_tactical_head import (
    RuleAwareTacticalHead,
    build_rule_aware_tactical_head_from_config,
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
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "rule_aware_tactical_head" in available_models()
    model = build_model("rule_aware_tactical_head", _toy_kwargs())
    assert isinstance(model, RuleAwareTacticalHead)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = RuleAwareTacticalHead(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["base_logit"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    assert out["primitive_gate"].shape == (2,)
    # gates are in [0, 1] by construction
    assert torch.all(out["primitive_gate"] >= 0.0)
    assert torch.all(out["primitive_gate"] <= 1.0)
    # all 11 raw TSDP features must be surfaced
    for name in TSDP_FEATURE_NAMES:
        key = f"tsdp_{name}"
        assert key in out
        assert out[key].shape == (2,)


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = RuleAwareTacticalHead(**_toy_kwargs())
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    out = model(boards)
    loss = out["logits"].pow(2).mean()
    loss.backward()
    # gradient must reach a trunk parameter and a head parameter
    trunk_param = next(model.trunk.parameters())
    head_param = next(model.delta_mlp.parameters())
    gate_param = next(model.gate_mlp.parameters())
    assert trunk_param.grad is not None
    assert head_param.grad is not None
    assert gate_param.grad is not None
    assert trunk_param.grad.abs().sum().item() > 0


def test_mate_position_features_are_correct() -> None:
    model = RuleAwareTacticalHead(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    # starting position must have mate_in_1 == 0
    assert out["tsdp_mate_in_1"][0].item() == 0.0
    # back-rank mate-in-1 must fire the indicator
    assert out["tsdp_mate_in_1"][1].item() == 1.0


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    cfg = dict(_toy_kwargs())
    model = RuleAwareTacticalHead(**cfg).eval()
    model.ablation = "zero_delta"
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_ablation_zeros_features_and_delta() -> None:
    model = RuleAwareTacticalHead(**_toy_kwargs(), ablation="trunk_only").eval()
    boards = _board_batch([
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    # delta must be zero
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    # raw features stay rule-correct even with ablation (they are raw)
    assert out["tsdp_mate_in_1"][0].item() == 1.0


def test_shuffle_ablation_keeps_logits_in_real_numbers() -> None:
    # we cannot make a deterministic assertion on shuffled outputs except that
    # the model returns finite logits and the right shape — the falsifier
    # itself runs at training time.
    torch.manual_seed(0)
    model = RuleAwareTacticalHead(**_toy_kwargs(), ablation="shuffle_tsdp").eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
        "k7/8/1Q2K3/8/8/8/8/8 w - - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    assert out["logits"].shape == (3,)


def test_disable_gate_holds_gate_at_one() -> None:
    model = RuleAwareTacticalHead(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))
    # delta = primitive_delta_raw exactly when gate=1
    assert torch.allclose(out["primitive_delta"], out["primitive_delta_raw"])


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,         # aliased -> trunk_channels
        "hidden_dim": 48,       # aliased -> trunk_hidden_dim
        "depth": 1,             # aliased -> trunk_depth
        "dropout": 0.0,         # aliased -> trunk_dropout
        "use_batchnorm": False, # aliased -> trunk_use_batchnorm
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_rule_aware_tactical_head_from_config(cfg)
    assert isinstance(model, RuleAwareTacticalHead)
    assert model.trunk.channels == 24


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        RuleAwareTacticalHead(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        RuleAwareTacticalHead(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        RuleAwareTacticalHead(input_channels=18, num_classes=3)
