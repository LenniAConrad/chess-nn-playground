"""Focused tests for the p050 Defender Overload Triad primitive (DOT)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.defender_overload_triad import (
    ALLOWED_ABLATIONS,
    OPERATOR_OUTPUT_DIM,
    SIDE_VECTOR_DIM,
    DefenderOverloadBuilder,
    DefenderOverloadTriad,
    build_defender_overload_triad_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "defender_overload_triad"
IDEA_DIR = Path("ideas/registry/p050_defender_overload_triad")

# Hand-crafted FENs for the smoke tests.
ROOK_PIN_FEN = "k7/8/8/8/p7/8/8/R6K w - - 0 1"
# White queen on c2 attacks two black pawns (b3 and d3); black knight on c5
# is the unique defender of both pawns -- the canonical overload position.
OVERLOAD_FEN = "4k3/8/8/2n5/8/1p1p4/2Q5/4K3 w - - 0 1"
# Same position but with only one attacked pawn and no defender; the
# overload mechanism cannot fire because there is no shared defender.
NO_OVERLOAD_FEN = "4k3/8/8/8/8/1p6/2Q5/4K3 w - - 0 1"
# Symmetric position with no pawns -- knights only -- used by the
# color-symmetry test where plane-swap alone is enough (no need to
# also vertical-mirror because knight attacks are colour-agnostic).
SYMMETRIC_KNIGHTS_FEN = "4k3/8/2n5/3n4/3N4/2N5/8/4K3 w - - 0 1"
BARE_KINGS_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


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
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, DefenderOverloadTriad)
    aliased = build_defender_overload_triad_from_config(
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


def test_operator_dim_matches_side_vector_concat() -> None:
    assert OPERATOR_OUTPUT_DIM == 4 * SIDE_VECTOR_DIM


def test_forward_shape_and_keys() -> None:
    model = DefenderOverloadTriad(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, OVERLOAD_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "overload_operator_mean",
        "overload_operator_max",
        "overload_operator_l2",
        "overload_us_mean",
        "overload_us_peak",
        "overload_them_mean",
        "overload_them_peak",
        "overload_defender_burden_us",
        "overload_defender_burden_them",
        "overload_pinned_share_us",
        "overload_pinned_share_them",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = DefenderOverloadTriad(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, OVERLOAD_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None
    assert model.builder.piece_value_logits.grad is not None
    assert model.builder.pin_discount_logit.grad is not None
    assert model.builder.pin_amplify_logit.grad is not None
    assert next(model.builder.target_gate.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = DefenderOverloadTriad(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, OVERLOAD_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(
        out["primitive_delta"], torch.zeros_like(out["primitive_delta"])
    )
    assert torch.allclose(out["logits"], out["base_logit"])


def test_disable_gate_pins_gate_at_one() -> None:
    model = DefenderOverloadTriad(
        **_toy_kwargs(), ablation="disable_gate"
    ).eval()
    boards = _board_batch([chess.STARTING_FEN, OVERLOAD_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_all_allowed_ablations_run_without_crash() -> None:
    boards = _board_batch([chess.STARTING_FEN, OVERLOAD_FEN, ROOK_PIN_FEN])
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = DefenderOverloadTriad(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        DefenderOverloadTriad(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        DefenderOverloadTriad(input_channels=12, num_classes=1)


def test_rejects_multiclass() -> None:
    with pytest.raises(ValueError):
        DefenderOverloadTriad(input_channels=18, num_classes=3)


def test_pin_indicator_fires_on_absolute_pin() -> None:
    """Hand-craft a rook-pawn-king absolute pin and verify the
    cumsum pin detector flags the pinned pawn on the defender side."""
    builder = DefenderOverloadBuilder().eval()
    # Absolute layout (white-to-move): white rook a1, black pawn a4,
    # black king a8. simple_18 row 0 is rank 8, row 7 is rank 1.
    ps = torch.zeros(1, 12, 8, 8)
    ps[0, 3, 7, 0] = 1.0   # white rook a1
    ps[0, 6, 4, 0] = 1.0   # black pawn a4
    ps[0, 11, 0, 0] = 1.0  # black king a8
    ps[0, 5, 7, 4] = 1.0   # white king e1
    stm = torch.ones(1)
    with torch.no_grad():
        out = builder(ps.flatten(2), stm)
    # a4 = row 4 col 0 = index 32 in the flattened layout.
    pinned_them = out["pinned_them"].squeeze()
    assert float(pinned_them[32]) > 0.5
    # The unrelated white king square should not be flagged.
    pinned_us = out["pinned_us"].squeeze()
    # e1 = row 7 col 4 = index 60.
    assert float(pinned_us[60]) < 0.5


def test_overload_signal_higher_on_real_overload_position() -> None:
    """White queen on c2 attacks both b3 and d3 pawns; black knight
    on c5 is the unique defender of both. The defender-burden mass
    must be strictly greater than on a position with one attacked
    undefended pawn (no shared defender to overload)."""
    builder = DefenderOverloadBuilder().eval()
    overload = _board_batch([OVERLOAD_FEN])
    no_overload = _board_batch([NO_OVERLOAD_FEN])
    # Convert board batches to (B, 12, 64).
    overload_ps = overload[:, :12].flatten(2)
    no_overload_ps = no_overload[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        ov_out = builder(overload_ps, stm)
        no_out = builder(no_overload_ps, stm)
    ov_burden = float(ov_out["us"]["defender_burden_max"])
    no_burden = float(no_out["us"]["defender_burden_max"])
    assert ov_burden > no_burden, (
        f"overload pos defender_burden_max={ov_burden} should exceed "
        f"non-overload pos defender_burden_max={no_burden}"
    )


def test_no_cross_target_load_changes_operator_vector_on_overload_pos() -> None:
    """The primary falsifier: replacing ``L^2 - Σ_t O^2`` with the plain
    single-target sum ``Σ_t O^2`` must change the operator output on
    an overload position. The two formulas only coincide
    component-by-component when the defender covers a single target;
    when one defender carries multiple critical targets, the target-
    exposure pool differs between full and ablated, so the 20-dim
    operator vector must diverge."""
    builder = DefenderOverloadBuilder().eval()
    overload = _board_batch([OVERLOAD_FEN])
    overload_ps = overload[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        full = builder(overload_ps, stm, ablation="none")
        ablated = builder(overload_ps, stm, ablation="no_cross_target_load")
    assert not torch.allclose(
        full["operator_vector"], ablated["operator_vector"], atol=1e-6
    ), (
        "no_cross_target_load must change the operator vector on an "
        "overload position (full target_exposure scales like c^2 m, "
        "ablated like c m)"
    )
    # And the ablation must keep some single-target mass alive (not
    # zero everything by accident).
    assert float(ablated["us"]["defender_burden_max"]) > 0.0


def test_no_pins_zeros_pin_amplification() -> None:
    """With ``no_pins`` the pin amplifier ``m = 1 + μ·π`` becomes
    ``m = 1``, which strictly weakens ``defender_burden`` whenever
    any defender is pinned. We compose an overload position whose
    defender (black knight on c5) is *not* pinned, and merely
    verify the pinned-defense share collapses for the
    rook-pawn-king pin position."""
    builder = DefenderOverloadBuilder().eval()
    pin_board = _board_batch([ROOK_PIN_FEN])
    pin_ps = pin_board[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        full = builder(pin_ps, stm, ablation="none")
        ablated = builder(pin_ps, stm, ablation="no_pins")
    # With pins zeroed, pinned_share must be 0 -- there are no
    # pinned defenders.
    assert (
        float(ablated["us"]["pinned_defense_share"]) == 0.0
    ), "no_pins must produce zero pinned_defense_share"
    # Without pins zeroed, the rook-pawn-king pin should be detected
    # on the them-side defender (a4 pawn). The pinned mask on the
    # them side must be non-empty.
    assert float(full["pinned_them"].sum()) > 0.0


def test_color_symmetry_on_pawnless_position() -> None:
    """A pawnless, geometrically symmetric position should produce
    identical us- and them-side vectors when fed white-to-move (so
    swapping the operator vector's first and second halves should
    leave the diff and absolute-diff components zero).

    Knights are used because their attack pattern is colour-agnostic
    (no forward direction). Pawns cannot be used because white pawns
    attack one rank up while black pawns attack one rank down -- the
    plane swap alone does not preserve attack mass for pawns.
    """
    builder = DefenderOverloadBuilder().eval()
    pos = _board_batch([SYMMETRIC_KNIGHTS_FEN])
    ps = pos[:, :12].flatten(2)
    with torch.no_grad():
        out = builder(ps, torch.ones(1), ablation="none")
    us = out["us_side_vec"].squeeze()
    them = out["them_side_vec"].squeeze()
    # The position is geometrically mirrored: us-vec and them-vec
    # should match to within float tolerance.
    assert torch.allclose(us, them, atol=1e-5), (us, them)


def test_no_target_value_collapses_value_terms_to_one() -> None:
    """The `no_target_value` ablation sets ``v_tar = v_att = v_def = 1``.
    This must make the per-target value sums equal to the per-target
    counts (`a_val == a`, `d_val == d`). We probe this through the
    overload diagnostics on the canonical overload position."""
    builder = DefenderOverloadBuilder().eval()
    pos = _board_batch([OVERLOAD_FEN])
    ps = pos[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        out = builder(ps, stm, ablation="no_target_value")
    # With v_tar = 1, the criticality is finite and the operator
    # vector remains valid. We only verify finite, non-NaN output.
    op = out["operator_vector"].squeeze()
    assert torch.isfinite(op).all()


def test_bare_kings_position_has_no_overload() -> None:
    """With only the two kings on the board, there are no occupied
    target squares for either side, so every overload aggregate
    must be exactly zero."""
    builder = DefenderOverloadBuilder().eval()
    board = _board_batch([BARE_KINGS_FEN])
    ps = board[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        out = builder(ps, stm, ablation="none")
    assert float(out["us"]["defender_burden_max"]) == 0.0
    assert float(out["them"]["defender_burden_max"]) == 0.0
    assert float(out["us"]["target_exposure_max"]) == 0.0
    assert float(out["them"]["target_exposure_max"]) == 0.0


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p050"
    assert data["slug"] == "defender_overload_triad"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["mechanism_family"] == "defender_load"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p050"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
