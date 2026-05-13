"""Focused tests for the p034 Octilinear Selective Scan primitive (OSS)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.octilinear_selective_scan import (
    ALLOWED_ABLATIONS,
    DIRECTION_STEPS,
    NUM_DIRECTIONS,
    OctilinearSelectiveScan,
    build_octilinear_selective_scan_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "octilinear_selective_scan"
IDEA_DIR = Path("ideas/registry/p034_octilinear_selective_scan")
ROOK_MATE_FEN = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"


def _board_batch(fens: list[str]) -> torch.Tensor:
    arrays = [fen_to_simple_18(fen) for fen in fens]
    return torch.from_numpy(np.stack(arrays, axis=0)).float()


def _toy_kwargs(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 32,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "feature_dim": 8,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, OctilinearSelectiveScan)
    aliased = build_octilinear_selective_scan_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "feature_dim": 6,
            "head_hidden_dim": 16,
        }
    )
    assert aliased.trunk.channels == 12


def test_direction_buffers_have_expected_shape() -> None:
    model = OctilinearSelectiveScan(**_toy_kwargs())
    for name, _, _ in DIRECTION_STEPS:
        tracks = getattr(model, f"tracks_{name}")
        # All tracks padded to length 8.
        assert tracks.shape[1] == 8, name
        # Cardinal directions have 8 tracks; diagonals have 15 tracks.
        if name in {"E", "W", "N", "S"}:
            assert tracks.shape[0] == 8, name
        else:
            assert tracks.shape[0] == 15, name


def test_forward_shape_and_keys() -> None:
    model = OctilinearSelectiveScan(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
    ):
        assert key in out and out[key].shape == (2,), key
    for name, _, _ in DIRECTION_STEPS:
        diag_key = f"oss_energy_{name}"
        assert diag_key in out and out[diag_key].shape == (2,), diag_key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = OctilinearSelectiveScan(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    for proj in model.a_logit_projections:
        assert proj.weight.grad is not None
    for proj in model.b_projections:
        assert proj.weight.grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = OctilinearSelectiveScan(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_single_direction_zeros_other_directions() -> None:
    model = OctilinearSelectiveScan(**_toy_kwargs(), ablation="single_direction").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    # Only the E direction (slot 0) should have nonzero energy; the other
    # 7 slots are zeroed in the forward path.
    e_energy = out["oss_energy_E"]
    assert torch.all(e_energy >= 0)
    for name, _, _ in DIRECTION_STEPS[1:]:
        diag = out[f"oss_energy_{name}"]
        assert torch.allclose(diag, torch.zeros_like(diag)), name


def test_fixed_transition_uses_data_independent_a() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = OctilinearSelectiveScan(**cfg).eval()
    torch.manual_seed(0)
    fixed = OctilinearSelectiveScan(**cfg, ablation="fixed_transition").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        fixed_out = fixed(boards)
    # Two paths should produce different deltas on at least one sample.
    assert not torch.allclose(
        full_out["primitive_delta_raw"], fixed_out["primitive_delta_raw"]
    )


def test_disable_gate_pins_gate_at_one() -> None:
    model = OctilinearSelectiveScan(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_shuffle_features_remains_finite() -> None:
    torch.manual_seed(0)
    model = OctilinearSelectiveScan(**_toy_kwargs(), ablation="shuffle_features").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        OctilinearSelectiveScan(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        OctilinearSelectiveScan(input_channels=12, num_classes=1)


def test_num_directions_matches_constant() -> None:
    assert NUM_DIRECTIONS == 8


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = OctilinearSelectiveScan(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p034"
    assert data["slug"] == "octilinear_selective_scan"
    assert data["implementation_kind"] == "bespoke_model"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p034"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
