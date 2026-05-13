"""Tests for the p023 Occlusion Semiring Delta-Bilinear Hyperedge primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.occlusion_semiring_delta_bilinear_hyperedge import (
    OPPOSITE_DIRECTION_PAIRS,
    OcclusionSemiringDeltaBilinearHyperedge,
    build_occlusion_semiring_delta_bilinear_hyperedge_from_config,
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
        "token_dim": 12,
        "hidden_dim": 16,
        "bilinear_dim": 8,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "occlusion_semiring_delta_bilinear_hyperedge" in available_models()
    model = build_model("occlusion_semiring_delta_bilinear_hyperedge", _toy_kwargs())
    assert isinstance(model, OcclusionSemiringDeltaBilinearHyperedge)


def test_pair_indices_are_geometric_opposites() -> None:
    # The opposite pairs must pair each direction with its 4-step rotation.
    assert OPPOSITE_DIRECTION_PAIRS == ((0, 4), (1, 5), (2, 6), (3, 7))


def test_forward_returns_required_keys_and_shapes() -> None:
    model = OcclusionSemiringDeltaBilinearHyperedge(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    for diag in ("osdb_hidden_magnitude", "osdb_pair_hyperedge_magnitude"):
        assert diag in out
        assert out[diag].shape == (2,)


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = OcclusionSemiringDeltaBilinearHyperedge(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad.abs().sum() > 0
    assert model.value_proj.weight.grad.abs().sum() > 0
    assert model.left_proj.weight.grad.abs().sum() > 0
    assert model.right_proj.weight.grad.abs().sum() > 0


def test_disable_bilinear_changes_delta_distribution() -> None:
    torch.manual_seed(0)
    model_bilinear = OcclusionSemiringDeltaBilinearHyperedge(**_toy_kwargs())
    torch.manual_seed(0)
    model_sum = OcclusionSemiringDeltaBilinearHyperedge(**_toy_kwargs(), ablation="disable_bilinear")
    for p1, p2 in zip(model_bilinear.parameters(), model_sum.parameters()):
        assert torch.equal(p1, p2)
    model_bilinear.eval(); model_sum.eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out_bi = model_bilinear(boards)
        out_sum = model_sum(boards)
    assert not torch.allclose(out_bi["primitive_delta_raw"], out_sum["primitive_delta_raw"])


def test_zero_occupancy_ablation_changes_hidden_magnitude() -> None:
    torch.manual_seed(0)
    model_full = OcclusionSemiringDeltaBilinearHyperedge(**_toy_kwargs())
    torch.manual_seed(0)
    model_zero = OcclusionSemiringDeltaBilinearHyperedge(**_toy_kwargs(), ablation="zero_occupancy")
    for p1, p2 in zip(model_full.parameters(), model_zero.parameters()):
        assert torch.equal(p1, p2)
    model_full.eval(); model_zero.eval()
    # Use a position with many blockers so the two ablations are differentiable.
    boards = _board_batch([
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ])
    with torch.no_grad():
        out_full = model_full(boards)
        out_zero = model_zero(boards)
    assert not torch.allclose(out_full["osdb_hidden_magnitude"], out_zero["osdb_hidden_magnitude"])


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = OcclusionSemiringDeltaBilinearHyperedge(**_toy_kwargs(), ablation="zero_delta").eval()
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
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "token_dim": 12,
        "hidden_dim": 18,
        "bilinear_dim": 8,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_occlusion_semiring_delta_bilinear_hyperedge_from_config(cfg)
    assert isinstance(model, OcclusionSemiringDeltaBilinearHyperedge)
    assert model.trunk.channels == 24


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        OcclusionSemiringDeltaBilinearHyperedge(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        OcclusionSemiringDeltaBilinearHyperedge(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        OcclusionSemiringDeltaBilinearHyperedge(input_channels=18, num_classes=3)
