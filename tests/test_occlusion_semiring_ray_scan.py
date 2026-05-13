"""Tests for the p021 Occlusion Semiring Ray Scan primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.occlusion_semiring_ray_scan import (
    OcclusionSemiringRayScan,
    _compute_transmittance,
    build_occlusion_semiring_ray_scan_from_config,
)
from chess_nn_playground.models.primitives.ray_geometry import RayGeometry
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
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "occlusion_semiring_ray_scan" in available_models()
    model = build_model("occlusion_semiring_ray_scan", _toy_kwargs())
    assert isinstance(model, OcclusionSemiringRayScan)


def test_empty_board_transmittance_is_one() -> None:
    g = RayGeometry.build()
    # Empty board: occupancy = 0 everywhere. Exclusive prefix product is 1 on
    # all valid steps and 0 off-board.
    occ = torch.zeros(1, 64)
    T = _compute_transmittance(occ, g.step_index, g.step_mask)
    mask = g.step_mask.unsqueeze(0)
    valid_T = T[mask > 0.5]
    assert torch.allclose(valid_T, torch.ones_like(valid_T))


def test_blocked_at_first_step_zeros_subsequent_transmittance() -> None:
    g = RayGeometry.build()
    # Build occupancy such that the first step on every ray from a1 is occupied.
    # In ray_geometry, from a1 (square 56) going N (direction 0), step indices are
    # [48, 40, 32, 24, 16, 8, 0]. We block square 48.
    occ = torch.zeros(1, 64)
    occ[0, 48] = 1.0
    T = _compute_transmittance(occ, g.step_index, g.step_mask)
    # For direction 0, source square 56, T at step 0 must be 1 (no previous step),
    # and at steps 1..6 must be 0 (blocked by square 48 at step 0).
    assert torch.isclose(T[0, 0, 56, 0], torch.tensor(1.0))
    for l in range(1, 7):
        assert T[0, 0, 56, l].item() < 1e-3, f"expected 0 at step {l}, got {T[0, 0, 56, l].item()}"


def test_forward_returns_required_keys_and_shapes() -> None:
    model = OcclusionSemiringRayScan(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    for diag in ("osrs_mean_transmittance", "osrs_open_ray_fraction"):
        assert diag in out
        assert out[diag].shape == (2,)
    assert torch.all(out["osrs_mean_transmittance"] >= 0.0)
    assert torch.all(out["osrs_mean_transmittance"] <= 1.0)


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = OcclusionSemiringRayScan(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    trunk_param = next(model.trunk.parameters())
    proj_param = model.direction_proj.weight
    assert trunk_param.grad is not None and trunk_param.grad.abs().sum() > 0
    assert proj_param.grad is not None and proj_param.grad.abs().sum() > 0


def test_zero_occupancy_ablation_makes_transmittance_constant() -> None:
    torch.manual_seed(0)
    model = OcclusionSemiringRayScan(**_toy_kwargs(), ablation="zero_occupancy").eval()
    boards = _board_batch([chess.STARTING_FEN, "k7/8/1Q2K3/8/8/8/8/8 w - - 0 1"])
    with torch.no_grad():
        out = model(boards)
    # Both samples have all ray cells fully visible -> mean transmittance is the
    # same across samples (ratio of valid steps to total step slots).
    assert torch.allclose(out["osrs_mean_transmittance"][0], out["osrs_mean_transmittance"][1])


def test_uniform_occupancy_ablation_zeros_deeper_steps() -> None:
    torch.manual_seed(0)
    model = OcclusionSemiringRayScan(**_toy_kwargs(), ablation="uniform_occupancy").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    # Mean transmittance must be < 1 / 7 (only step 0 is non-zero).
    assert out["osrs_mean_transmittance"][0].item() < 0.20


def test_isotropic_A_ablation_keeps_finite_logits() -> None:
    torch.manual_seed(0)
    model = OcclusionSemiringRayScan(**_toy_kwargs(), ablation="isotropic_A").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = OcclusionSemiringRayScan(**_toy_kwargs(), ablation="zero_delta").eval()
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
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_occlusion_semiring_ray_scan_from_config(cfg)
    assert isinstance(model, OcclusionSemiringRayScan)
    assert model.trunk.channels == 24


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        OcclusionSemiringRayScan(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        OcclusionSemiringRayScan(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        OcclusionSemiringRayScan(input_channels=18, num_classes=3)
