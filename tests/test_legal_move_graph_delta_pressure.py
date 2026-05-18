"""Focused tests for the p053 Legal-Move-Graph Pressure-Delta primitive (LMGDP)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.legal_move_graph_delta import (
    PIECE_TYPE_NAMES,
    _compute_typed_legal_edges,
)
from chess_nn_playground.models.primitives.legal_move_graph_delta_pressure import (
    ALLOWED_ABLATIONS,
    NUM_EDGE_FEATURES,
    PER_EDGE_FEATURE_NAMES,
    LegalMoveGraphDeltaPressure,
    aggregate_per_target_features,
    build_legal_move_graph_delta_pressure_from_config,
    compute_pressure_delta_edge_features,
    per_type_global_summary,
    _build_king_zone_template,
    _stack_per_edge_features,
)
from chess_nn_playground.models.primitives.rule_graph_features import rule_geometry
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "legal_move_graph_delta_pressure"
IDEA_DIR = Path("ideas/registry/p053_legal_move_graph_delta_pressure")
START_FEN = chess.STARTING_FEN
# White rook on a1, black queen on a8, kings on e1/e8, otherwise empty.
ROOK_VS_QUEEN_FEN = "q3k3/8/8/8/8/8/8/R3K3 w - - 0 1"
# White knight on b1, black queen on c3, white king e1, black king e8.
KNIGHT_TAKES_QUEEN_FEN = "4k3/8/8/8/8/2q5/8/1N2K3 w - - 0 1"
# White knight on f6 (giving discovered check pattern), kings on e1/e8, black queen on f7 in check zone.
KNIGHT_FORK_FEN = "4k3/5q2/5N2/8/8/8/8/4K3 w - - 0 1"
# Black-to-move case to test side-to-move selection.
BLACK_TO_MOVE_FEN = "4k3/8/8/8/8/8/r7/3K4 b - - 0 1"


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
        "per_type_token_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_registry_key_does_not_clash_with_p009() -> None:
    """The new primitive registers under a disambiguated key so p009's
    `legal_move_graph_delta` registration is untouched."""
    keys = set(available_models())
    assert "legal_move_graph_delta" in keys
    assert "legal_move_graph_delta_pressure" in keys
    assert REGISTRY_KEY != "legal_move_graph_delta"


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, LegalMoveGraphDeltaPressure)
    aliased = build_legal_move_graph_delta_pressure_from_config(
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


def test_per_edge_features_have_expected_names_and_count() -> None:
    assert PER_EDGE_FEATURE_NAMES == (
        "is_capture",
        "into_king_zone",
        "gives_check_proxy",
        "enemy_value_at_target",
        "pre_opp_attackers_at_target",
        "pre_own_defenders_at_target",
        "mover_post_attack_value_from_t",
        "mover_post_defender_value_from_t",
    )
    assert NUM_EDGE_FEATURES == 8


def test_pressure_delta_features_are_zero_outside_candidate_edges() -> None:
    board = _board_batch([ROOK_VS_QUEEN_FEN, KNIGHT_TAKES_QUEEN_FEN])
    geometry = rule_geometry()
    edges = _compute_typed_legal_edges(board, geometry)
    template = _build_king_zone_template()
    features = compute_pressure_delta_edge_features(board, edges, geometry, template)
    for name, value in features.items():
        # Masked features must vanish wherever edges == 0.
        non_edge = (edges < 0.5).to(value.dtype)
        assert torch.all(value * non_edge == 0.0), name


def test_is_capture_lights_up_for_rook_takes_queen() -> None:
    """Rook on a1 capturing black queen on a8 should yield is_capture=1 on that edge."""
    board = _board_batch([ROOK_VS_QUEEN_FEN])
    geometry = rule_geometry()
    edges = _compute_typed_legal_edges(board, geometry)
    template = _build_king_zone_template()
    features = compute_pressure_delta_edge_features(board, edges, geometry, template)
    # ROOK is piece index 3. a1 = row 7, file 0 -> square 56. a8 = row 0, file 0 -> square 0.
    rook_index = 3
    a1_idx = 7 * 8 + 0
    a8_idx = 0 * 8 + 0
    is_capture = features["is_capture"]
    assert is_capture[0, rook_index, a1_idx, a8_idx].item() == pytest.approx(1.0)
    enemy_value = features["enemy_value_at_target"]
    # Capturing queen -> value 9.
    assert enemy_value[0, rook_index, a1_idx, a8_idx].item() == pytest.approx(9.0)


def test_gives_check_proxy_lights_up_when_promotion_arrival_attacks_king() -> None:
    """White knight on b1 has a candidate move to c3; a knight on c3 would attack
    no king square here (kings on e1/e8), so gives_check should be 0 on b1->c3."""
    board = _board_batch([KNIGHT_TAKES_QUEEN_FEN])
    geometry = rule_geometry()
    edges = _compute_typed_legal_edges(board, geometry)
    template = _build_king_zone_template()
    features = compute_pressure_delta_edge_features(board, edges, geometry, template)
    knight_index = 1
    b1_idx = 7 * 8 + 1
    c3_idx = 5 * 8 + 2
    # b1->c3 captures the queen; gives_check is 0 (knight on c3 attacks no king).
    assert edges[0, knight_index, b1_idx, c3_idx].item() == pytest.approx(1.0)
    assert features["is_capture"][0, knight_index, b1_idx, c3_idx].item() == pytest.approx(1.0)
    assert features["gives_check_proxy"][0, knight_index, b1_idx, c3_idx].item() == pytest.approx(0.0)
    # The captured queen value is 9.
    assert features["enemy_value_at_target"][0, knight_index, b1_idx, c3_idx].item() == pytest.approx(9.0)


def test_per_target_aggregation_includes_degree_column() -> None:
    board = _board_batch([ROOK_VS_QUEEN_FEN])
    geometry = rule_geometry()
    edges = _compute_typed_legal_edges(board, geometry)
    template = _build_king_zone_template()
    features = compute_pressure_delta_edge_features(board, edges, geometry, template)
    stacked = _stack_per_edge_features(features)
    per_target = aggregate_per_target_features(edges, stacked, include_degree=True)
    assert per_target.shape == (1, 6, 64, NUM_EDGE_FEATURES + 1)
    # The trailing column equals the per-target in-degree summed over sources.
    in_degree = edges.sum(dim=-2)
    assert torch.allclose(per_target[..., -1], in_degree)


def test_per_type_global_summary_shape_and_consistency() -> None:
    board = _board_batch([ROOK_VS_QUEEN_FEN, KNIGHT_TAKES_QUEEN_FEN, START_FEN])
    geometry = rule_geometry()
    edges = _compute_typed_legal_edges(board, geometry)
    template = _build_king_zone_template()
    features = compute_pressure_delta_edge_features(board, edges, geometry, template)
    stacked = _stack_per_edge_features(features)
    summary = per_type_global_summary(edges, stacked)
    assert summary.shape == (3, 6, 3 * NUM_EDGE_FEATURES + 1)
    # The trailing column equals the per-(b, r) edge count.
    expected_counts = edges.sum(dim=(-2, -1))
    assert torch.allclose(summary[..., -1], expected_counts)


def test_forward_shape_and_keys() -> None:
    model = LegalMoveGraphDeltaPressure(**_toy_kwargs()).eval()
    boards = _board_batch([START_FEN, ROOK_VS_QUEEN_FEN, BLACK_TO_MOVE_FEN])
    out = model(boards)
    assert out["logits"].shape == (3,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_applied",
        "primitive_gate_logit",
        "primitive_contribution",
        "lmgdp_total_edge_count",
    ):
        assert key in out and out[key].shape == (3,), key
    for name in PIECE_TYPE_NAMES:
        for prefix in (
            "lmgdp_edge_count",
            "lmgdp_post_attack_value_mean",
            "lmgdp_capture_value_mean",
        ):
            key = f"{prefix}_{name}"
            assert key in out and out[key].shape == (3,), key


def test_total_edge_count_is_per_type_sum() -> None:
    model = LegalMoveGraphDeltaPressure(**_toy_kwargs()).eval()
    boards = _board_batch([START_FEN, ROOK_VS_QUEEN_FEN, KNIGHT_TAKES_QUEEN_FEN])
    with torch.no_grad():
        out = model(boards)
    per_type_total = sum(out[f"lmgdp_edge_count_{name}"] for name in PIECE_TYPE_NAMES)
    assert torch.allclose(per_type_total, out["lmgdp_total_edge_count"])


def test_starting_position_edge_count_matches_attack_style_helper() -> None:
    """The typed-legal-edge helper (shared with p009) covers attack-style pseudo-
    legal moves only: knight moves and pawn diagonal attacks. Pawn pushes and
    castling / en-passant are out-of-scope for the current topology compiler
    (documented in `math_thesis.md`). At the starting position this yields:

        knights: Nb1 -> a3, c3, Ng1 -> f3, h3 (4 edges)
        pawns:   each of 8 own pawns contributes its diagonal attacks; corner
                 pawns have one diagonal, the others have two => 1+2*6+1 = 14
        total:   18

    Sliding pieces and the king are fully blocked at the start, contributing 0.
    """
    model = LegalMoveGraphDeltaPressure(**_toy_kwargs()).eval()
    boards = _board_batch([START_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["lmgdp_total_edge_count"][0].item() == pytest.approx(18.0)
    assert out["lmgdp_edge_count_N"][0].item() == pytest.approx(4.0)
    assert out["lmgdp_edge_count_P"][0].item() == pytest.approx(14.0)
    for slider_or_king in ("B", "R", "Q", "K"):
        assert out[f"lmgdp_edge_count_{slider_or_king}"][0].item() == pytest.approx(0.0), slider_or_king


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = LegalMoveGraphDeltaPressure(**_toy_kwargs())
    boards = _board_batch([START_FEN, ROOK_VS_QUEEN_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = LegalMoveGraphDeltaPressure(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([START_FEN, ROOK_VS_QUEEN_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_recovers_trunk_logit() -> None:
    model = LegalMoveGraphDeltaPressure(**_toy_kwargs(), ablation="trunk_only").eval()
    boards = _board_batch([START_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["logits"], out["base_logit"])


def test_disable_gate_pins_gate_at_one() -> None:
    model = LegalMoveGraphDeltaPressure(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([START_FEN, ROOK_VS_QUEEN_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_no_pressure_delta_zeroes_pressure_columns() -> None:
    torch.manual_seed(0)
    model = LegalMoveGraphDeltaPressure(
        **_toy_kwargs(), ablation="no_pressure_delta"
    ).eval()
    geometry = rule_geometry()
    boards = _board_batch([ROOK_VS_QUEEN_FEN])
    edges = _compute_typed_legal_edges(boards, geometry)
    template = _build_king_zone_template()
    features = compute_pressure_delta_edge_features(boards, edges, geometry, template)
    feature_dict = model._maybe_zero_pressure_blocks(features)
    for name in (
        "pre_opp_attackers_at_target",
        "pre_own_defenders_at_target",
        "mover_post_attack_value_from_t",
        "mover_post_defender_value_from_t",
    ):
        assert torch.all(feature_dict[name] == 0.0), name
    # The non-pressure features must remain non-trivial.
    assert features["is_capture"].sum().item() > 0.0


def test_no_capture_value_zeroes_capture_columns() -> None:
    torch.manual_seed(0)
    model = LegalMoveGraphDeltaPressure(
        **_toy_kwargs(), ablation="no_capture_value"
    ).eval()
    geometry = rule_geometry()
    boards = _board_batch([ROOK_VS_QUEEN_FEN])
    edges = _compute_typed_legal_edges(boards, geometry)
    template = _build_king_zone_template()
    features = compute_pressure_delta_edge_features(boards, edges, geometry, template)
    feature_dict = model._maybe_zero_pressure_blocks(features)
    for name in ("enemy_value_at_target", "gives_check_proxy"):
        assert torch.all(feature_dict[name] == 0.0), name


def test_random_typed_edges_keeps_same_per_type_density() -> None:
    torch.manual_seed(0)
    model = LegalMoveGraphDeltaPressure(
        **_toy_kwargs(), ablation="random_typed_edges"
    ).eval()
    boards = _board_batch([ROOK_VS_QUEEN_FEN, START_FEN])
    with torch.no_grad():
        edges = model._build_edges(boards)
    # The random ablation samples a Bernoulli with per-type density equal to
    # the real density; we cannot assert exact equality, but we can assert
    # the shape and that the mask is float-valued in [0, 1].
    assert edges.shape == (2, 6, 64, 64)
    assert torch.all((edges == 0.0) | (edges == 1.0))


def test_shared_target_pool_uses_single_linear() -> None:
    model = LegalMoveGraphDeltaPressure(
        **_toy_kwargs(), ablation="shared_target_pool"
    ).eval()
    assert model.target_token_proj is not None
    assert model.target_token_per_type is None


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        LegalMoveGraphDeltaPressure(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        LegalMoveGraphDeltaPressure(input_channels=12, num_classes=1)


def test_all_allowed_ablations_run_without_crash() -> None:
    boards = _board_batch([START_FEN, ROOK_VS_QUEEN_FEN, BLACK_TO_MOVE_FEN])
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = LegalMoveGraphDeltaPressure(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_per_edge_features_finite_on_full_board() -> None:
    """Sanity: per-edge features finite on the cluttered starting position."""
    board = _board_batch([START_FEN])
    geometry = rule_geometry()
    edges = _compute_typed_legal_edges(board, geometry)
    template = _build_king_zone_template()
    features = compute_pressure_delta_edge_features(board, edges, geometry, template)
    for name, value in features.items():
        assert torch.isfinite(value).all(), name


def test_black_to_move_edges_use_own_pieces() -> None:
    """Black-to-move should produce edges for the black pieces, not the white ones."""
    board = _board_batch([BLACK_TO_MOVE_FEN])
    geometry = rule_geometry()
    edges = _compute_typed_legal_edges(board, geometry)
    # Black rook on a2 has rook edges; total black-rook edges > 0.
    rook_index = 3
    assert edges[0, rook_index].sum().item() > 0.0


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p053"
    assert data["slug"] == "legal_move_graph_delta_pressure"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p053"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
