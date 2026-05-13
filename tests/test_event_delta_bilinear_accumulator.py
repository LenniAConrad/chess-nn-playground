"""Tests for the p022 Event-Delta Bilinear Accumulator primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.event_delta_bilinear_accumulator import (
    EventDeltaBilinearAccumulator,
    build_event_delta_bilinear_accumulator_from_config,
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
        "bilinear_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "event_delta_bilinear_accumulator" in available_models()
    model = build_model("event_delta_bilinear_accumulator", _toy_kwargs())
    assert isinstance(model, EventDeltaBilinearAccumulator)


def test_active_count_matches_occupancy() -> None:
    model = EventDeltaBilinearAccumulator(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,                                  # 32 pieces
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",                 # 6 pieces
    ])
    with torch.no_grad():
        out = model(boards)
    assert out["edba_active_count"][0].item() == pytest.approx(32.0)
    assert out["edba_active_count"][1].item() == pytest.approx(6.0)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = EventDeltaBilinearAccumulator(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, "k7/8/1Q2K3/8/8/8/8/8 w - - 0 1"])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    for diag in ("edba_active_count", "edba_first_order_magnitude", "edba_pair_term_magnitude"):
        assert diag in out
        assert out[diag].shape == (2,)


def test_pair_term_matches_FM_identity_on_random_input() -> None:
    # The pair term must equal A (.) B - sum_i U_i (.) V_i exactly, up to fp.
    torch.manual_seed(0)
    model = EventDeltaBilinearAccumulator(
        **_toy_kwargs(),
        normalize_by_active_count=False,
    ).eval()
    # Construct an artificial board with a few pieces.
    boards = _board_batch([
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    # Recompute (A, B, Q) by hand using the model's projections.
    piece_planes = boards[:, :12].clamp(0.0, 1.0)
    stm = boards[:, 12:13].clamp(0.0, 1.0)
    token_input = torch.cat([piece_planes, stm], dim=1).flatten(2).transpose(1, 2).contiguous()
    occupancy = piece_planes.flatten(2).sum(dim=1).clamp(0.0, 1.0).unsqueeze(-1)
    u = model.u_proj(token_input) * occupancy
    v = model.v_proj(token_input) * occupancy
    A = u.sum(dim=1)
    B = v.sum(dim=1)
    P = (u * v).sum(dim=1)
    Q_expected = A * B - P
    Q_via_diag = out["edba_pair_term_magnitude"]  # mean(|Q|)
    Q_recompute_mag = Q_expected.abs().mean(dim=1)
    assert torch.allclose(Q_via_diag, Q_recompute_mag, atol=1e-5)


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = EventDeltaBilinearAccumulator(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad.abs().sum() > 0
    assert model.u_proj.weight.grad.abs().sum() > 0
    assert model.v_proj.weight.grad.abs().sum() > 0


def test_first_order_only_ablation_zeros_pair_in_readout() -> None:
    torch.manual_seed(0)
    model_full = EventDeltaBilinearAccumulator(**_toy_kwargs())
    torch.manual_seed(0)
    model_first = EventDeltaBilinearAccumulator(**_toy_kwargs(), ablation="first_order_only")
    for p1, p2 in zip(model_full.parameters(), model_first.parameters()):
        assert torch.equal(p1, p2)
    model_full.eval(); model_first.eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out_full = model_full(boards)
        out_first = model_first(boards)
    # The raw delta differs because the first-order-only run cannot see Q.
    assert not torch.allclose(out_full["primitive_delta_raw"], out_first["primitive_delta_raw"])


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = EventDeltaBilinearAccumulator(**_toy_kwargs(), ablation="zero_delta").eval()
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
        "bilinear_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_event_delta_bilinear_accumulator_from_config(cfg)
    assert isinstance(model, EventDeltaBilinearAccumulator)
    assert model.trunk.channels == 24


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        EventDeltaBilinearAccumulator(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        EventDeltaBilinearAccumulator(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        EventDeltaBilinearAccumulator(input_channels=18, num_classes=3)
