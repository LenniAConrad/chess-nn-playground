"""Focused tests for the p052 Promotion and Underpromotion Geometry primitive (PUGP)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.promotion_underpromotion import (
    ALLOWED_ABLATIONS,
    PROMOTION_TYPE_NAMES,
    PromotionUnderpromotionGeometry,
    build_promotion_underpromotion_from_config,
    canonicalize_simple_18,
    compute_per_file_candidates,
    compute_promoted_attack_features,
    sliding_attack_masks_from_arrival,
    _build_canonicalize_perm,
)
from chess_nn_playground.models.primitives.ray_geometry import build_ray_step_index
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "promotion_underpromotion"
IDEA_DIR = Path("ideas/registry/p052_promotion_underpromotion")
START_FEN = chess.STARTING_FEN
# A white-to-move quiet promotion: white pawn on a7, board otherwise empty besides kings.
WHITE_QUIET_PROMOTION_FEN = "8/P7/8/8/8/8/8/k6K w - - 0 1"
# White-to-move pawn on h7 with a black bishop on g8 (capture-left promotion to g8 available).
WHITE_CAP_LEFT_PROMOTION_FEN = "6b1/7P/8/8/8/8/8/k6K w - - 0 1"
# White-to-move pawn on a7 with a black knight on b8 (capture-right promotion).
WHITE_CAP_RIGHT_PROMOTION_FEN = "1n6/P7/8/8/8/8/8/k6K w - - 0 1"
# Black-to-move quiet promotion: black pawn on a2.
BLACK_QUIET_PROMOTION_FEN = "k6K/8/8/8/8/8/p7/8 b - - 0 1"
# A position with no near-promotion pawns.
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
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, PromotionUnderpromotionGeometry)
    aliased = build_promotion_underpromotion_from_config(
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


def test_canonicalize_is_identity_for_white_to_move() -> None:
    perm = _build_canonicalize_perm()
    board = _board_batch([WHITE_QUIET_PROMOTION_FEN])
    canonical = canonicalize_simple_18(board, perm)
    # STM was 1.0 already; the canonical STM plane should remain a constant 1.0.
    assert torch.allclose(canonical[:, 12], torch.ones_like(canonical[:, 12]))
    # Own pawn was at white-pawn plane 0; canonical own-pawn plane is still 0.
    # The pawn on a7 (rank 7 = plane row 1, file 0) survives the identity.
    assert canonical[0, 0, 1, 0].item() == pytest.approx(1.0)


def test_canonicalize_for_black_to_move_brings_own_pawn_to_row_one() -> None:
    perm = _build_canonicalize_perm()
    board = _board_batch([BLACK_QUIET_PROMOTION_FEN])
    canonical = canonicalize_simple_18(board, perm)
    # STM plane fixed to 1.0.
    assert torch.allclose(canonical[:, 12], torch.ones_like(canonical[:, 12]))
    # Black pawn was on a2 (plane row 6, file 0, plane 6). After flip + swap
    # it should appear as own-pawn (canonical plane 0) at row 1, file 0.
    assert canonical[0, 0, 1, 0].item() == pytest.approx(1.0)


def test_canonicalize_is_involutive_on_piece_planes() -> None:
    perm = _build_canonicalize_perm()
    board = _board_batch([WHITE_CAP_LEFT_PROMOTION_FEN, BLACK_QUIET_PROMOTION_FEN])
    canonical = canonicalize_simple_18(board, perm)
    # The canonical board ALWAYS reports white-to-move. Applying canonicalize a
    # second time should therefore be a no-op on the canonical board.
    canonical_twice = canonicalize_simple_18(canonical, perm)
    assert torch.allclose(canonical, canonical_twice)


def test_per_file_candidates_quiet_push_detected() -> None:
    perm = _build_canonicalize_perm()
    board = _board_batch([WHITE_QUIET_PROMOTION_FEN])
    canonical = canonicalize_simple_18(board, perm)
    candidates = compute_per_file_candidates(canonical)
    push = candidates["push_mask"]
    assert push.shape == (1, 8)
    assert push[0, 0].item() == pytest.approx(1.0)
    # Other files have no near-promotion pawn.
    for f in range(1, 8):
        assert push[0, f].item() == pytest.approx(0.0)
    # No capture candidates in this empty arrival-rank position.
    assert candidates["capL_mask"].sum().item() == pytest.approx(0.0)
    assert candidates["capR_mask"].sum().item() == pytest.approx(0.0)


def test_per_file_candidates_capture_left_detected() -> None:
    perm = _build_canonicalize_perm()
    board = _board_batch([WHITE_CAP_LEFT_PROMOTION_FEN])
    canonical = canonicalize_simple_18(board, perm)
    candidates = compute_per_file_candidates(canonical)
    # Source pawn is on h7 (file 7). The capture-left target g8 (file 6) has an
    # enemy bishop, so capL_mask[7] == 1. capR is blocked off-board.
    assert candidates["capL_mask"][0, 7].item() == pytest.approx(1.0)
    assert candidates["capR_mask"].sum().item() == pytest.approx(0.0)
    # Push to h8 is blocked? No -- h8 is empty in this FEN. So push[7] == 1 too.
    assert candidates["push_mask"][0, 7].item() == pytest.approx(1.0)


def test_per_file_candidates_capture_right_detected() -> None:
    perm = _build_canonicalize_perm()
    board = _board_batch([WHITE_CAP_RIGHT_PROMOTION_FEN])
    canonical = canonicalize_simple_18(board, perm)
    candidates = compute_per_file_candidates(canonical)
    # Source pawn is on a7 (file 0). Capture-right target b8 (file 1) has an
    # enemy knight, so capR_mask[0] == 1.
    assert candidates["capR_mask"][0, 0].item() == pytest.approx(1.0)
    assert candidates["capL_mask"].sum().item() == pytest.approx(0.0)


def test_per_file_candidates_quiet_blocked_by_own_piece_above() -> None:
    perm = _build_canonicalize_perm()
    # White pawn on a7 with a white rook on a8 -- quiet push is blocked.
    blocked_fen = "R7/P7/8/8/8/8/8/k6K w - - 0 1"
    board = _board_batch([blocked_fen])
    canonical = canonicalize_simple_18(board, perm)
    candidates = compute_per_file_candidates(canonical)
    assert candidates["push_mask"][0, 0].item() == pytest.approx(0.0)


def test_per_file_candidates_starting_position_has_no_promotion() -> None:
    perm = _build_canonicalize_perm()
    board = _board_batch([START_FEN])
    canonical = canonicalize_simple_18(board, perm)
    candidates = compute_per_file_candidates(canonical)
    assert candidates["push_mask"].sum().item() == pytest.approx(0.0)
    assert candidates["capL_mask"].sum().item() == pytest.approx(0.0)
    assert candidates["capR_mask"].sum().item() == pytest.approx(0.0)


def test_sliding_attack_mask_from_open_corner_matches_queen_attacks() -> None:
    """A queen on a8 of an otherwise empty board attacks 21 squares."""
    perm = _build_canonicalize_perm()
    empty_kings_fen = "8/8/8/8/8/8/8/k6K w - - 0 1"
    board = _board_batch([empty_kings_fen])
    canonical = canonicalize_simple_18(board, perm)
    occupancy_flat = canonical[:, :12].sum(dim=1).clamp(0.0, 1.0).reshape(1, -1)
    # Mask out the king squares so only "queen on a8" is considered.
    occupancy_flat = occupancy_flat * 0.0
    ri, rm = build_ray_step_index()
    slide_attack = sliding_attack_masks_from_arrival(occupancy_flat, ri, rm)
    # slide_attack[b, dirs, srcs, dst]. Queen attack from src=0 is sum over dirs.
    queen_attack_from_a8 = slide_attack[0, :, 0, :].sum(dim=0).clamp(0.0, 1.0)
    # Self square is never attacked by sliding from itself.
    assert queen_attack_from_a8[0].item() == pytest.approx(0.0)
    # Empty board: queen on a8 attacks 7 (row 0 across) + 7 (file 0 down) + 7 (a8-h1 diagonal) = 21 squares.
    assert queen_attack_from_a8.sum().item() == pytest.approx(21.0)


def test_promoted_attack_check_indicator_fires_for_promotion_to_check() -> None:
    """Queen promotion to a8 with the enemy king on h8 should give check on rank 8."""
    perm = _build_canonicalize_perm()
    # White pawn on a7 about to promote; black king on h8. Empty otherwise.
    fen = "7k/P7/8/8/8/8/8/7K w - - 0 1"
    board = _board_batch([fen])
    canonical = canonicalize_simple_18(board, perm)
    ri, rm = build_ray_step_index()
    from chess_nn_playground.models.primitives.promotion_underpromotion import (
        _build_knight_attack_template,
        _build_king_zone_template,
    )
    knight = _build_knight_attack_template()
    king_zone = _build_king_zone_template()
    attacks = compute_promoted_attack_features(canonical, ri, rm, knight, king_zone)
    # Queen index 0; arrival square index 0 (file 0, canonical row 0).
    queen_check_a8 = attacks["check"][0, 0, 0].item()
    assert queen_check_a8 == pytest.approx(1.0)
    # Knight from a8 does NOT attack h8 (it would attack b6, c7 -- not h8).
    knight_check_a8 = attacks["check"][0, 3, 0].item()
    assert knight_check_a8 == pytest.approx(0.0)


def test_forward_shape_and_keys() -> None:
    model = PromotionUnderpromotionGeometry(**_toy_kwargs()).eval()
    boards = _board_batch([START_FEN, WHITE_QUIET_PROMOTION_FEN, BLACK_QUIET_PROMOTION_FEN])
    out = model(boards)
    assert out["logits"].shape == (3,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "pugp_push_count",
        "pugp_capL_count",
        "pugp_capR_count",
        "pugp_total_count",
        "pugp_n_own_r1",
        "pugp_n_opp_r1",
        "pugp_knight_fork_max",
        "pugp_queen_check_count",
        "pugp_queen_zone_max",
    ):
        assert key in out and out[key].shape == (3,), key


def test_total_count_matches_canonical_candidates() -> None:
    model = PromotionUnderpromotionGeometry(**_toy_kwargs()).eval()
    boards = _board_batch(
        [
            START_FEN,
            WHITE_QUIET_PROMOTION_FEN,
            WHITE_CAP_LEFT_PROMOTION_FEN,
            BLACK_QUIET_PROMOTION_FEN,
            ROOK_MATE_FEN,
        ]
    )
    with torch.no_grad():
        out = model(boards)
    total = out["pugp_total_count"]
    # START_FEN: no candidates.
    assert total[0].item() == pytest.approx(0.0)
    # WHITE_QUIET: one push candidate.
    assert total[1].item() == pytest.approx(1.0)
    # WHITE_CAP_LEFT: push + capL = 2.
    assert total[2].item() == pytest.approx(2.0)
    # BLACK_QUIET: canonicalised symmetric to WHITE_QUIET = 1 push candidate.
    assert total[3].item() == pytest.approx(1.0)
    # ROOK_MATE: no near-promotion own pawn.
    assert total[4].item() == pytest.approx(0.0)


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = PromotionUnderpromotionGeometry(**_toy_kwargs())
    boards = _board_batch([START_FEN, WHITE_CAP_LEFT_PROMOTION_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = PromotionUnderpromotionGeometry(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([START_FEN, WHITE_QUIET_PROMOTION_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_disable_gate_pins_gate_at_one() -> None:
    model = PromotionUnderpromotionGeometry(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([START_FEN, WHITE_QUIET_PROMOTION_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_pseudo_only_changes_candidate_counts() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = PromotionUnderpromotionGeometry(**cfg).eval()
    torch.manual_seed(0)
    pseudo = PromotionUnderpromotionGeometry(**cfg, ablation="pseudo_only").eval()
    # WHITE_QUIET_PROMOTION_FEN: a7 pawn with empty a8 -> both full and pseudo
    # count 1 push. To see a divergence, build a position where the quiet
    # arrival is blocked (so full sees 0 push, pseudo sees 1).
    blocked_fen = "R7/P7/8/8/8/8/8/k6K w - - 0 1"
    boards = _board_batch([blocked_fen])
    with torch.no_grad():
        full_out = full(boards)
        pseudo_out = pseudo(boards)
    assert full_out["pugp_push_count"][0].item() == pytest.approx(0.0)
    assert pseudo_out["pugp_push_count"][0].item() == pytest.approx(1.0)


def test_no_capture_zeros_capture_counts() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    nocap = PromotionUnderpromotionGeometry(**cfg, ablation="no_capture").eval()
    boards = _board_batch([WHITE_CAP_LEFT_PROMOTION_FEN])
    with torch.no_grad():
        out = nocap(boards)
    assert out["pugp_capL_count"][0].item() == pytest.approx(0.0)
    assert out["pugp_capR_count"][0].item() == pytest.approx(0.0)


def test_no_promotion_pawn_yields_zero_total_count() -> None:
    """No own near-promotion pawn must yield zero PUGP candidate counts everywhere."""
    model = PromotionUnderpromotionGeometry(**_toy_kwargs()).eval()
    boards = _board_batch([START_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["pugp_total_count"], torch.zeros_like(out["pugp_total_count"]))


def test_promoted_attacks_finite_under_full_board() -> None:
    """Sanity: promoted attack features finite on the cluttered starting position."""
    perm = _build_canonicalize_perm()
    board = _board_batch([START_FEN])
    canonical = canonicalize_simple_18(board, perm)
    ri, rm = build_ray_step_index()
    from chess_nn_playground.models.primitives.promotion_underpromotion import (
        _build_knight_attack_template,
        _build_king_zone_template,
    )
    knight = _build_knight_attack_template()
    king_zone = _build_king_zone_template()
    attacks = compute_promoted_attack_features(canonical, ri, rm, knight, king_zone)
    for key, value in attacks.items():
        assert torch.isfinite(value).all(), key


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        PromotionUnderpromotionGeometry(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        PromotionUnderpromotionGeometry(input_channels=12, num_classes=1)


def test_all_allowed_ablations_run_without_crash() -> None:
    boards = _board_batch([START_FEN, WHITE_CAP_LEFT_PROMOTION_FEN, BLACK_QUIET_PROMOTION_FEN])
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = PromotionUnderpromotionGeometry(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_promotion_type_names_match_pfct_order() -> None:
    assert PROMOTION_TYPE_NAMES == ("Q", "R", "B", "N")


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p052"
    assert data["slug"] == "promotion_underpromotion"
    assert data["implementation_kind"] == "bespoke_model"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p052"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
