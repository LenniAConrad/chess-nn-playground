"""Tests for the p037 Gibbs Cut Log-Partition Operator primitive."""

from __future__ import annotations

import chess
import math
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.gibbs_cut_log_partition import (
    GibbsCutLogPartition,
    _build_state_bits,
    _build_within_row_xor,
    _build_between_row_xor,
    build_gibbs_cut_log_partition_from_config,
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
        "grid_h": 3,
        "grid_w": 3,
        "d_cut": 2,
        "temperature": 1.0,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "gibbs_cut_log_partition" in available_models()
    model = build_model("gibbs_cut_log_partition", _toy_kwargs())
    assert isinstance(model, GibbsCutLogPartition)


def test_state_bits_shape_and_values() -> None:
    bits = _build_state_bits(4)
    assert bits.shape == (16, 4)
    # Check bit 0 alternates each row.
    expected_bit0 = torch.tensor([float(k & 1) for k in range(16)])
    assert torch.equal(bits[:, 0], expected_bit0)


def test_within_xor_marks_horizontal_boundary() -> None:
    xor = _build_within_row_xor(3)
    assert xor.shape == (8, 2)
    # State 0b010 = 2 should have xor[0]=(0 vs 1)=1, xor[1]=(1 vs 0)=1.
    state = 0b010
    assert xor[state, 0].item() == 1.0
    assert xor[state, 1].item() == 1.0
    # State 0b111 = 7 has no boundary -> xor = [0, 0].
    assert xor[7, 0].item() == 0.0
    assert xor[7, 1].item() == 0.0


def test_between_xor_tensor_shape() -> None:
    xor = _build_between_row_xor(3)
    assert xor.shape == (8, 8, 3)


def test_brute_force_log_partition_matches_dp() -> None:
    # Compare the row-transfer DP to a brute-force sum over all
    # subsets for a tiny 2x2 grid.
    torch.manual_seed(0)
    model = GibbsCutLogPartition(**_toy_kwargs(grid_h=2, grid_w=2, d_cut=1, temperature=1.0)).eval()
    H, W, d, tau = 2, 2, 1, 1.0
    B = 1
    c_h = torch.rand(B, H, W - 1, d) + 0.1
    c_v = torch.rand(B, H - 1, W, d) + 0.1
    s = torch.rand(B, H, W, d) + 0.1
    t = torch.rand(B, H, W, d) + 0.1
    with torch.no_grad():
        dp = model._compute_log_partition(c_h, c_v, s, t)  # (B, d)

    # Brute force over all 2^4 = 16 subsets.
    flat_idx = torch.arange(16)
    log_Z = torch.tensor(-math.inf)
    for k in range(16):
        # Decode S as a (2, 2) binary occupancy.
        bits_flat = torch.tensor([(k >> j) & 1 for j in range(4)], dtype=torch.float32)
        S = bits_flat.view(H, W)
        energy = 0.0
        # Horizontal edges (within row)
        for r in range(H):
            for j in range(W - 1):
                energy += float(c_h[0, r, j, 0]) * abs(float(S[r, j] - S[r, j + 1]))
        # Vertical edges (between rows)
        for r in range(H - 1):
            for j in range(W):
                energy += float(c_v[0, r, j, 0]) * abs(float(S[r, j] - S[r + 1, j]))
        # Cell penalties
        for r in range(H):
            for j in range(W):
                if S[r, j] == 0:
                    energy += float(s[0, r, j, 0])
                else:
                    energy += float(t[0, r, j, 0])
        log_Z = torch.logaddexp(log_Z, torch.tensor(-energy / tau))

    assert torch.isclose(dp[0, 0], log_Z, atol=1e-4), (dp[0, 0].item(), log_Z.item())


def test_forward_returns_required_keys_and_shapes() -> None:
    model = GibbsCutLogPartition(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    assert out["logits"].shape == (1,)
    for diag in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "gibbs_log_partition_mean",
        "gibbs_log_partition_max",
        "gibbs_cut_edge_energy",
    ):
        assert diag in out


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = GibbsCutLogPartition(**_toy_kwargs(ablation="zero_delta")).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_uniform_edges_ablation_changes_log_partition() -> None:
    torch.manual_seed(0)
    model_full = GibbsCutLogPartition(**_toy_kwargs())
    torch.manual_seed(0)
    model_uniform = GibbsCutLogPartition(**_toy_kwargs(ablation="uniform_edges"))
    for p1, p2 in zip(model_full.parameters(), model_uniform.parameters()):
        assert torch.equal(p1, p2)
    model_full.eval(); model_uniform.eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out_full = model_full(boards)
        out_uniform = model_uniform(boards)
    assert not torch.allclose(
        out_full["gibbs_log_partition_mean"], out_uniform["gibbs_log_partition_mean"]
    )


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = GibbsCutLogPartition(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad.abs().sum() > 0
    assert model.edge_h_proj.weight.grad.abs().sum() > 0
    assert model.source_proj.weight.grad.abs().sum() > 0


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "grid_h": 3,
        "grid_w": 3,
        "d_cut": 2,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_gibbs_cut_log_partition_from_config(cfg)
    assert isinstance(model, GibbsCutLogPartition)
    assert model.trunk.channels == 24


def test_rejects_grid_w_too_large() -> None:
    with pytest.raises(ValueError):
        GibbsCutLogPartition(**_toy_kwargs(grid_w=7))


def test_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        GibbsCutLogPartition(**_toy_kwargs(temperature=0.0))


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        GibbsCutLogPartition(**_toy_kwargs(ablation="not_a_real_ablation"))


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        GibbsCutLogPartition(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        GibbsCutLogPartition(input_channels=18, num_classes=3)
