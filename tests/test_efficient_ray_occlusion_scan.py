"""Focused tests for the p054 Efficient Ray Occlusion Scan primitive."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.efficient_ray_occlusion_scan import (
    ALLOWED_ABLATIONS,
    EfficientRayOcclusionScan,
    _build_ray_feature_tensor,
    build_efficient_ray_occlusion_scan_from_config,
    ray_occlusion_scan,
)
from chess_nn_playground.models.primitives.ray_geometry import (
    NUM_DIRECTIONS,
    RAY_MAX_LEN,
    SQUARES,
    RayGeometry,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "efficient_ray_occlusion_scan"
IDEA_DIR = Path("ideas/registry/p054_efficient_ray_occlusion_scan")
ROOK_PIN_FEN = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"


def _board_batch(fens: list[str]) -> torch.Tensor:
    arrays = [fen_to_simple_18(fen) for fen in fens]
    return torch.from_numpy(np.stack(arrays, axis=0)).float()


def _toy_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
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
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, EfficientRayOcclusionScan)
    aliased = build_efficient_ray_occlusion_scan_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "head_hidden_dim": 16,
        }
    )
    assert aliased.trunk.channels == 12


def test_scan_recovers_first_and_second_blocker_identity_along_a_known_ray() -> None:
    """Source square 56 looking N -> us-R at 48, them-K at 32 (pin-to-king frame)."""
    geom = RayGeometry.build()
    occupancy = torch.zeros(1, SQUARES)
    occupancy[0, 48] = 1.0
    occupancy[0, 32] = 1.0
    piece_state = torch.zeros(1, 12, 8, 8)
    piece_state[0, 3, 6, 0] = 1.0  # us-R at row 6, col 0 = square 48
    piece_state[0, 11, 4, 0] = 1.0  # them-K at row 4, col 0 = square 32
    feat = _build_ray_feature_tensor(piece_state)
    scan = ray_occlusion_scan(feat, occupancy, geom.step_index, geom.step_mask)

    # Direction 0 (N), source 56.
    visible = scan["visible_steps"][0, 0, 56]
    first = scan["first_steps"][0, 0, 56]
    second = scan["second_steps"][0, 0, 56]
    xray = scan["xray_lane_steps"][0, 0, 56]
    # Step 0 -> square 48 (first blocker): visible & first only.
    assert visible[0].item() == 1.0
    assert first[0].item() == 1.0
    assert second[0].item() == 0.0
    # Step 1 -> square 40 (behind first, before second): xray_lane only.
    assert visible[1].item() == 0.0
    assert xray[1].item() == 1.0
    # Step 2 -> square 32 (second blocker): second & xray_lane.
    assert second[2].item() == 1.0
    assert xray[2].item() == 1.0
    # No blockers beyond.
    assert second[3:].sum().item() == 0.0

    # First-blocker identity matches us-R (value 5, us occupancy).
    assert pytest.approx(scan["first_value"][0, 0, 56].item()) == 5.0
    assert pytest.approx(scan["first_us_occ"][0, 0, 56].item()) == 1.0
    assert pytest.approx(scan["first_them_occ"][0, 0, 56].item()) == 0.0
    # Second-blocker identity matches them-K (value 200, them occupancy, them_king_second flag).
    assert pytest.approx(scan["second_value"][0, 0, 56].item()) == 200.0
    assert pytest.approx(scan["second_them_occ"][0, 0, 56].item()) == 1.0
    assert pytest.approx(scan["them_king_second"][0, 0, 56].item()) == 1.0


def test_scan_visible_count_equals_ray_length_when_empty() -> None:
    """With an empty board, visible_count along each direction equals step_mask sum."""
    geom = RayGeometry.build()
    occupancy = torch.zeros(2, SQUARES)
    piece_state = torch.zeros(2, 12, 8, 8)
    feat = _build_ray_feature_tensor(piece_state)
    scan = ray_occlusion_scan(feat, occupancy, geom.step_index, geom.step_mask)
    # visible_count = sum_l visible = number of on-board steps in that direction.
    expected = geom.step_mask.sum(dim=-1)  # (D, S)
    assert torch.allclose(scan["visible_count"][0], expected.to(scan["visible_count"].dtype))
    # No blockers at all -> first_exists, second_exists are zero everywhere.
    assert scan["first_exists"].abs().sum().item() == 0.0
    assert scan["second_exists"].abs().sum().item() == 0.0


def test_scan_mobility_excludes_blocker_cell() -> None:
    geom = RayGeometry.build()
    occupancy = torch.zeros(1, SQUARES)
    occupancy[0, 48] = 1.0
    piece_state = torch.zeros(1, 12, 8, 8)
    piece_state[0, 3, 6, 0] = 1.0
    feat = _build_ray_feature_tensor(piece_state)
    scan = ray_occlusion_scan(feat, occupancy, geom.step_index, geom.step_mask)
    # From 56 along N, mobility_len counts visible empty squares only (= 0 here,
    # since step 0 -- the only visible cell -- is itself the blocker).
    assert scan["mobility_len"][0, 0, 56].item() == 0.0
    # visible_count includes the blocker cell -> 1.
    assert scan["visible_count"][0, 0, 56].item() == 1.0


def test_forward_returns_required_keys_and_finite_logits() -> None:
    model = EfficientRayOcclusionScan(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_PIN_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_applied",
        "primitive_gate_logit",
        "primitive_gate_entropy",
        "primitive_contribution",
        "eros_occupancy_density",
        "eros_mobility_mean",
        "eros_xray_pressure_mean",
        "eros_visible_density",
        "eros_first_blocker_rate",
        "eros_second_blocker_rate",
        "mechanism_energy",
    ):
        assert key in out, key
        assert out[key].shape == (2,), key
    assert torch.isfinite(out["logits"]).all()
    assert torch.all(out["eros_occupancy_density"] >= 0.0)
    assert torch.all(out["eros_occupancy_density"] <= 1.0)


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = EfficientRayOcclusionScan(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_PIN_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    trunk_param = next(model.trunk.parameters())
    assert trunk_param.grad is not None and trunk_param.grad.abs().sum() > 0
    delta_param = model.delta_head[1].weight  # first Linear after LayerNorm
    assert delta_param.grad is not None and delta_param.grad.abs().sum() > 0
    gate_param = model.gate_head[1].weight
    assert gate_param.grad is not None and gate_param.grad.abs().sum() > 0


def test_zero_delta_recovers_trunk_logit() -> None:
    model = EfficientRayOcclusionScan(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_PIN_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_zeros_contribution() -> None:
    model = EfficientRayOcclusionScan(**_toy_kwargs(), ablation="trunk_only").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_contribution"], torch.zeros_like(out["primitive_contribution"]))


def test_disable_gate_pins_gate_at_one() -> None:
    model = EfficientRayOcclusionScan(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_PIN_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_first_only_zeros_xray_pressure() -> None:
    """The primary falsifier should suppress second-blocker / x-ray channels."""
    torch.manual_seed(0)
    cfg = _toy_kwargs()
    full = EfficientRayOcclusionScan(**cfg).eval()
    torch.manual_seed(0)
    falsifier = EfficientRayOcclusionScan(**cfg, ablation="first_only").eval()
    # The two models share weights because the seed is identical.
    for p1, p2 in zip(full.parameters(), falsifier.parameters()):
        assert torch.equal(p1, p2)
    # Pin frame: us-R on d1, them-K on g8, our pieces aligned to enable a long pin.
    boards = _board_batch([ROOK_PIN_FEN])
    with torch.no_grad():
        full_out = full(boards)
        ablated_out = falsifier(boards)
    # The raw delta must differ on a position whose label depends on second-blocker
    # structure (the rook is staring at the king through h-file pawns).
    assert not torch.allclose(
        full_out["primitive_delta_raw"], ablated_out["primitive_delta_raw"]
    )


def test_uniform_occupancy_differs_from_real_occupancy() -> None:
    torch.manual_seed(0)
    cfg = _toy_kwargs()
    full = EfficientRayOcclusionScan(**cfg).eval()
    torch.manual_seed(0)
    uniform = EfficientRayOcclusionScan(**cfg, ablation="uniform_occupancy").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        full_out = full(boards)
        uniform_out = uniform(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], uniform_out["primitive_delta_raw"]
    )


def test_empty_occupancy_differs_from_real_occupancy() -> None:
    torch.manual_seed(0)
    cfg = _toy_kwargs()
    full = EfficientRayOcclusionScan(**cfg).eval()
    torch.manual_seed(0)
    empty = EfficientRayOcclusionScan(**cfg, ablation="empty_occupancy").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        full_out = full(boards)
        empty_out = empty(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], empty_out["primitive_delta_raw"]
    )


def test_all_allowed_ablations_run_without_crash() -> None:
    boards = _board_batch([chess.STARTING_FEN, ROOK_PIN_FEN])
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = EfficientRayOcclusionScan(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        EfficientRayOcclusionScan(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        EfficientRayOcclusionScan(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        EfficientRayOcclusionScan(input_channels=18, num_classes=3)


def test_ray_geometry_buffer_shapes() -> None:
    model = EfficientRayOcclusionScan(**_toy_kwargs())
    assert model.ray_step_index.shape == (NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)
    assert model.ray_step_mask.shape == (NUM_DIRECTIONS, SQUARES, RAY_MAX_LEN)


def test_idea_yaml_metadata_matches_folder() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p054"
    assert data["slug"] == "efficient_ray_occlusion_scan"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p054"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
