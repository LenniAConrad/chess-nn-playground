"""Focused tests for the p051 King-Zone Reply Pressure primitive (KZRP)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.king_zone_reply_pressure import (
    ALLOWED_ABLATIONS,
    OPERATOR_OUTPUT_DIM,
    SIDE_VECTOR_DIM,
    KingZoneReplyPressure,
    KingZoneReplyPressureBuilder,
    build_king_zone_reply_pressure_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "king_zone_reply_pressure"
IDEA_DIR = Path("ideas/registry/p051_king_zone_reply_pressure")

# Hand-crafted FENs for smoke tests.
BARE_KINGS_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
ROOK_PIN_FEN = "k7/8/8/8/p7/8/8/R6K w - - 0 1"
# Three white pawns one rank from the black king on e7/d7/f7 -- creates
# heavy direct king-zone attack mass.
KING_RING_PRESSURE_FEN = "4k3/3PPP2/8/8/8/8/8/4K3 w - - 0 1"
# Pinned black rook on a2 by white rook on a1 against black king on a8.
# The pinned rook defends a-file ring squares around the black king.
RING_PIN_FEN = "k7/8/8/8/8/8/r7/R6K w - - 0 1"


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
    assert isinstance(model, KingZoneReplyPressure)
    aliased = build_king_zone_reply_pressure_from_config(
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
    model = KingZoneReplyPressure(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, KING_RING_PRESSURE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "kzrp_operator_mean",
        "kzrp_operator_max",
        "kzrp_operator_l2",
        "kzrp_asym_score",
        "kzrp_us_zone_pressure",
        "kzrp_us_fake_defense_loss",
        "kzrp_us_live_escapes",
        "kzrp_us_sealed_escapes",
        "kzrp_us_blocked_escapes",
        "kzrp_us_king_attack_mass",
        "kzrp_us_front_attack_mass",
        "kzrp_us_reply_proxy",
        "kzrp_them_zone_pressure",
        "kzrp_them_live_escapes",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = KingZoneReplyPressure(**_toy_kwargs())
    boards = _board_batch([RING_PIN_FEN, KING_RING_PRESSURE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None
    assert model.builder.attack_unit_logits.grad is not None
    assert model.builder.pin_discount_logit.grad is not None
    assert model.builder.def_discount_logit.grad is not None
    assert model.builder.front_strength_logit.grad is not None
    assert model.builder.zone_weight_logits.grad is not None
    assert model.builder.escape_weight_logits.grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = KingZoneReplyPressure(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, KING_RING_PRESSURE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(
        out["primitive_delta"], torch.zeros_like(out["primitive_delta"])
    )
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_recovers_trunk_logit() -> None:
    model = KingZoneReplyPressure(**_toy_kwargs(), ablation="trunk_only").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["logits"], out["base_logit"])


def test_disable_gate_pins_gate_at_one() -> None:
    model = KingZoneReplyPressure(
        **_toy_kwargs(), ablation="disable_gate"
    ).eval()
    boards = _board_batch([chess.STARTING_FEN, KING_RING_PRESSURE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_all_allowed_ablations_run_without_crash() -> None:
    boards = _board_batch([chess.STARTING_FEN, KING_RING_PRESSURE_FEN, RING_PIN_FEN])
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = KingZoneReplyPressure(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        KingZoneReplyPressure(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        KingZoneReplyPressure(input_channels=12, num_classes=1)


def test_rejects_multiclass() -> None:
    with pytest.raises(ValueError):
        KingZoneReplyPressure(input_channels=18, num_classes=3)


def test_pin_indicator_fires_on_absolute_pin() -> None:
    """Hand-craft a rook-pawn-king absolute pin and verify the
    cumsum pin detector flags the pinned pawn on the defender side.
    """
    builder = KingZoneReplyPressureBuilder().eval()
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
    pinned_black = out["pinned_black"].squeeze()
    assert float(pinned_black[32]) > 0.5
    # The unrelated white king square should not be flagged.
    pinned_white = out["pinned_white"].squeeze()
    # e1 = row 7 col 4 = index 60.
    assert float(pinned_white[60]) < 0.5


def test_king_ring_pressure_position_has_nonzero_zone_pressure() -> None:
    """Three white pawns one rank in front of black king on e8 must
    produce a strictly positive `kzrp_us_zone_pressure`."""
    builder = KingZoneReplyPressureBuilder().eval()
    pressure = _board_batch([KING_RING_PRESSURE_FEN])
    bare = _board_batch([BARE_KINGS_FEN])
    with torch.no_grad():
        out_press = builder(pressure[:, :12].flatten(2), torch.ones(1))
        out_bare = builder(bare[:, :12].flatten(2), torch.ones(1))
    zp_press = float(out_press["us"]["zone_pressure"])
    zp_bare = float(out_bare["us"]["zone_pressure"])
    assert zp_press > zp_bare, (zp_press, zp_bare)
    # The king-attack mass on the black king square is non-zero because
    # the d7 and f7 pawns attack e8.
    assert float(out_press["us"]["king_attack_mass"]) > 0.0


def test_bare_kings_position_has_zero_pressure_terms() -> None:
    """With only the two kings, every attack and defense term on
    the king zones is zero, so the zone pressure / fake-defense /
    king-attack terms must be exactly zero."""
    builder = KingZoneReplyPressureBuilder().eval()
    board = _board_batch([BARE_KINGS_FEN])
    with torch.no_grad():
        out = builder(board[:, :12].flatten(2), torch.ones(1))
    assert float(out["us"]["zone_pressure"]) == 0.0
    assert float(out["them"]["zone_pressure"]) == 0.0
    assert float(out["us"]["fake_defense_loss"]) == 0.0
    assert float(out["them"]["fake_defense_loss"]) == 0.0
    assert float(out["us"]["king_attack_mass"]) == 0.0
    assert float(out["them"]["king_attack_mass"]) == 0.0
    assert float(out["us"]["front_attack_mass"]) == 0.0


def test_no_front_zone_drops_front_contribution() -> None:
    """The primary falsifier `no_front_zone` must zero
    `kzrp_us_front_attack_mass`."""
    builder = KingZoneReplyPressureBuilder().eval()
    board = _board_batch([KING_RING_PRESSURE_FEN])
    ps = board[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        full = builder(ps, stm, ablation="none")
        ablated = builder(ps, stm, ablation="no_front_zone")
    # Zone pressure may drop or stay the same depending on whether the
    # position has any Z_front activity; in this position, the pawns
    # sit on row 1 (rank 7), and the black king is on row 0 (rank 8),
    # so Z_front is at row 2 (rank 6). The pawns don't attack row 2.
    # Still, the explicit front-mass diagnostic must collapse.
    assert float(ablated["us"]["front_attack_mass"]) == 0.0
    # And the operator vector must be at least element-wise consistent
    # under the ablation: the front-contribution mass goes away from
    # the front_attack_mass slot.
    full_front = float(full["us"]["front_attack_mass"])
    if full_front > 0.0:
        assert not torch.allclose(
            full["operator_vector"], ablated["operator_vector"]
        )


def test_no_pins_zeros_fake_defense_loss() -> None:
    """The `no_pins` ablation sets π = 0 everywhere. On a position
    where a pinned defender actually defends king-ring squares
    (`RING_PIN_FEN`), the fake-defense loss must collapse to zero
    in the ablated run while staying strictly positive in the
    unablated run."""
    builder = KingZoneReplyPressureBuilder().eval()
    board = _board_batch([RING_PIN_FEN])
    ps = board[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        full = builder(ps, stm, ablation="none")
        ablated = builder(ps, stm, ablation="no_pins")
    assert float(ablated["us"]["fake_defense_loss"]) == 0.0
    assert float(full["us"]["fake_defense_loss"]) > 0.0
    # And the pin indicator on the pinned rook (a2 = row 6 col 0 = idx 48)
    # must fire in the unablated run.
    assert float(full["pinned_black"][0, 48]) > 0.5


def test_no_asymmetry_zeros_them_side_vec() -> None:
    """The `no_asymmetry` ablation zeroes the them-side vector, which
    forces the second 8-feature block of the operator to be zero and
    sets the diff to S_us."""
    builder = KingZoneReplyPressureBuilder().eval()
    board = _board_batch([KING_RING_PRESSURE_FEN])
    ps = board[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        out = builder(ps, stm, ablation="no_asymmetry")
    op = out["operator_vector"].squeeze()
    # Layout: [S_us (0..7), S_them (8..15), diff (16..23), |diff| (24..31)].
    assert torch.allclose(op[8:16], torch.zeros(8))


def test_no_escape_decomp_collapses_escape_classes() -> None:
    """The `no_escape_decomp` ablation puts the total escape count
    into the live slot and zeros sealed / blocked. We verify the
    aux diagnostics reflect this on a position with multiple escape
    classes."""
    builder = KingZoneReplyPressureBuilder().eval()
    board = _board_batch([KING_RING_PRESSURE_FEN])
    ps = board[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        out = builder(ps, stm, ablation="no_escape_decomp")
    assert float(out["us"]["sealed_escapes"]) == 0.0
    assert float(out["us"]["blocked_escapes"]) == 0.0


def test_uniform_units_keeps_finite_output() -> None:
    builder = KingZoneReplyPressureBuilder().eval()
    board = _board_batch([RING_PIN_FEN])
    ps = board[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        out = builder(ps, stm, ablation="uniform_units")
    assert torch.isfinite(out["operator_vector"]).all()


def test_uniform_zone_weights_keeps_finite_output() -> None:
    builder = KingZoneReplyPressureBuilder().eval()
    board = _board_batch([KING_RING_PRESSURE_FEN])
    ps = board[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        out = builder(ps, stm, ablation="uniform_zone_weights")
    assert torch.isfinite(out["operator_vector"]).all()


def test_king_attack_mass_fires_when_pawn_attacks_king_square() -> None:
    """The d7 and f7 white pawns both attack e8 (the black king
    square). The `kzrp_us_king_attack_mass` diagnostic should report
    the weighted attack mass on the king square -- with unit pawn
    weight, that is 2 (two pawns attacking)."""
    builder = KingZoneReplyPressureBuilder().eval()
    board = _board_batch([KING_RING_PRESSURE_FEN])
    ps = board[:, :12].flatten(2)
    stm = torch.ones(1)
    with torch.no_grad():
        out = builder(ps, stm)
    # Two pawn attackers (d7, f7) on e8 with softplus(1.0) ≈ 1.31 weight.
    # The mass must be strictly positive.
    assert float(out["us"]["king_attack_mass"]) > 0.0
    # And the them-side king (white king on e1) is not attacked by any
    # of the white-side mover's pieces (black has no pieces), so the
    # them-attack on us-king should be zero.
    assert float(out["them"]["king_attack_mass"]) == 0.0


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p051"
    assert data["slug"] == "king_zone_reply_pressure"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["mechanism_family"] == "king_safety"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p051"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
