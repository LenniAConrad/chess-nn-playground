"""Tests for the p024 Event-Symmetric Interaction Accumulator primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.event_symmetric_interaction_accumulator import (
    EventSymmetricInteractionAccumulator,
    build_event_symmetric_interaction_accumulator_from_config,
    compute_elementary_symmetric,
)
from chess_nn_playground.models.registry import available_models, build_model


def _toy_kwargs(order: int = 2) -> dict[str, object]:
    return {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 32,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "token_dim": 12,
        "order": order,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "event_symmetric_interaction_accumulator" in available_models()
    model = build_model("event_symmetric_interaction_accumulator", _toy_kwargs())
    assert isinstance(model, EventSymmetricInteractionAccumulator)


def test_elementary_symmetric_first_order_equals_sum() -> None:
    # E^{(1)} must equal sum_i u_i.
    torch.manual_seed(0)
    B, D = 2, 8
    tokens = torch.randn(B, 64, D)
    occupancy = torch.zeros(B, 64)
    # Mark 3 squares active per sample.
    occupancy[:, [0, 5, 10]] = 1.0
    e_list = compute_elementary_symmetric(tokens, occupancy, order=1)
    expected = (tokens * occupancy.unsqueeze(-1)).sum(dim=1)
    assert torch.allclose(e_list[0], expected)


def test_elementary_symmetric_second_order_equals_FM_identity() -> None:
    # E^{(2)} must equal (1/2) * (S1 (.) S1 - sum_i u_i (.) u_i).
    torch.manual_seed(0)
    B, D = 2, 8
    tokens = torch.randn(B, 64, D)
    occupancy = torch.zeros(B, 64)
    occupancy[:, [1, 4, 9, 17]] = 1.0
    e_list = compute_elementary_symmetric(tokens, occupancy, order=2)
    masked = tokens * occupancy.unsqueeze(-1)
    s1 = masked.sum(dim=1)
    s2_diag = (masked * masked).sum(dim=1)
    expected_e2 = 0.5 * (s1 * s1 - s2_diag)
    assert torch.allclose(e_list[1], expected_e2, atol=1e-5)


def test_active_count_matches_occupancy() -> None:
    model = EventSymmetricInteractionAccumulator(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    assert out["esia_active_count"][0].item() == pytest.approx(32.0)
    assert out["esia_active_count"][1].item() == pytest.approx(6.0)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = EventSymmetricInteractionAccumulator(**_toy_kwargs(order=2)).eval()
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    assert out["logits"].shape == (1,)
    for diag in ("esia_active_count", "esia_order_max_magnitude", "esia_order_mean_magnitude"):
        assert diag in out
    assert "esia_order_1_magnitude" in out
    assert "esia_order_2_magnitude" in out


def test_order_3_works_and_exposes_three_diagnostics() -> None:
    model = EventSymmetricInteractionAccumulator(**_toy_kwargs(order=3)).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert "esia_order_1_magnitude" in out
    assert "esia_order_2_magnitude" in out
    assert "esia_order_3_magnitude" in out
    assert torch.isfinite(out["logits"]).all()


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = EventSymmetricInteractionAccumulator(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad.abs().sum() > 0
    assert model.token_proj.weight.grad.abs().sum() > 0


def test_first_order_only_ablation_changes_delta_distribution() -> None:
    torch.manual_seed(0)
    model_full = EventSymmetricInteractionAccumulator(**_toy_kwargs())
    torch.manual_seed(0)
    model_first = EventSymmetricInteractionAccumulator(**_toy_kwargs(), ablation="first_order_only")
    for p1, p2 in zip(model_full.parameters(), model_first.parameters()):
        assert torch.equal(p1, p2)
    model_full.eval(); model_first.eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out_full = model_full(boards)
        out_first = model_first(boards)
    assert not torch.allclose(out_full["primitive_delta_raw"], out_first["primitive_delta_raw"])


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = EventSymmetricInteractionAccumulator(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "token_dim": 12,
        "order": 2,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_event_symmetric_interaction_accumulator_from_config(cfg)
    assert isinstance(model, EventSymmetricInteractionAccumulator)
    assert model.trunk.channels == 24


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        EventSymmetricInteractionAccumulator(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_invalid_order() -> None:
    with pytest.raises(ValueError):
        EventSymmetricInteractionAccumulator(**_toy_kwargs(order=4))


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        EventSymmetricInteractionAccumulator(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        EventSymmetricInteractionAccumulator(input_channels=18, num_classes=3)
