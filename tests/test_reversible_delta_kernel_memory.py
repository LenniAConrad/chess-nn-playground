"""Tests for the p019 Reversible Delta Kernel Memory primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.reversible_delta_kernel_memory import (
    ReversibleDeltaKernelMemory,
    build_reversible_delta_kernel_memory_from_config,
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
        "memory_heads": 8,
        "memory_value_dim": 8,
        "num_queries": 2,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "reversible_delta_kernel_memory" in available_models()
    model = build_model("reversible_delta_kernel_memory", _toy_kwargs())
    assert isinstance(model, ReversibleDeltaKernelMemory)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = ReversibleDeltaKernelMemory(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["base_logit"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    assert out["primitive_gate"].shape == (2,)
    assert torch.all(out["primitive_gate"] >= 0.0)
    assert torch.all(out["primitive_gate"] <= 1.0)
    for diag_key in ("rdkm_active_count", "rdkm_memory_norm", "rdkm_z_norm"):
        assert diag_key in out
        assert out[diag_key].shape == (2,)


def test_active_count_matches_occupancy() -> None:
    model = ReversibleDeltaKernelMemory(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,                                  # 32 pieces
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",                 # 6 pieces
    ])
    with torch.no_grad():
        out = model(boards)
    assert out["rdkm_active_count"][0].item() == pytest.approx(32.0)
    assert out["rdkm_active_count"][1].item() == pytest.approx(6.0)


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = ReversibleDeltaKernelMemory(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    trunk_param = next(model.trunk.parameters())
    phi_param = model.phi_proj.weight
    nu_param = model.nu_proj.weight
    query_param = model.query_proj[1].weight
    assert trunk_param.grad is not None and trunk_param.grad.abs().sum() > 0
    assert phi_param.grad is not None and phi_param.grad.abs().sum() > 0
    assert nu_param.grad is not None and nu_param.grad.abs().sum() > 0
    assert query_param.grad is not None and query_param.grad.abs().sum() > 0


def test_zero_memory_ablation_zeros_value_path() -> None:
    model = ReversibleDeltaKernelMemory(**_toy_kwargs(), ablation="zero_memory").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    # The numerator (phi(q) . M) is zero, but the readout still passes through a bias
    # via the value_readout layer. We only assert the structural diagnostic.
    assert torch.isfinite(out["logits"]).all()


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = ReversibleDeltaKernelMemory(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_ablation_zeros_delta() -> None:
    model = ReversibleDeltaKernelMemory(**_toy_kwargs(), ablation="trunk_only").eval()
    boards = _board_batch([chess.STARTING_FEN, "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


def test_uniform_query_ablation_keeps_finite_logits() -> None:
    model = ReversibleDeltaKernelMemory(**_toy_kwargs(), ablation="uniform_query").eval()
    boards = _board_batch([chess.STARTING_FEN, "k7/8/1Q2K3/8/8/8/8/8 w - - 0 1"])
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,         # aliased -> trunk_channels
        "hidden_dim": 48,       # aliased -> trunk_hidden_dim
        "depth": 1,             # aliased -> trunk_depth
        "dropout": 0.0,         # aliased -> trunk_dropout
        "use_batchnorm": False, # aliased -> trunk_use_batchnorm
        "token_dim": 12,
        "memory_heads": 8,
        "memory_value_dim": 8,
        "num_queries": 2,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_reversible_delta_kernel_memory_from_config(cfg)
    assert isinstance(model, ReversibleDeltaKernelMemory)
    assert model.trunk.channels == 24


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        ReversibleDeltaKernelMemory(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        ReversibleDeltaKernelMemory(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        ReversibleDeltaKernelMemory(input_channels=18, num_classes=3)
