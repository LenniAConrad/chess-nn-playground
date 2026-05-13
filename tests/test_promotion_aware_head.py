"""Focused tests for the Promotion-Aware Head primitive (idea i246, PFCT)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.promotion_aware_head import (
    ALLOWED_ABLATIONS,
    PROMOTION_TYPE_NAMES,
    PROMOTION_TYPE_COUNT,
    PromotionAwareHead,
    build_promotion_aware_head_from_config,
    build_promotion_counterfactuals,
    find_near_promotion_slots,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "promotion_aware_head"
IDEA_DIR = Path("ideas/registry/i246_promotion_aware_head")

# White king on e1, black king on e8, white pawn on a7 (white-to-move). The
# pawn is one rank from promotion; the position is a textbook PFCT trigger.
WHITE_NEAR_PROMOTION_FEN = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
# Black king on e8, white king on e1, black pawn on a2 (black-to-move).
BLACK_NEAR_PROMOTION_FEN = "4k3/8/8/8/8/8/p7/4K3 b - - 0 1"
# Standard initial position — no near-promotion pawns.
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Two white pawns on the 7th rank (a7, h7), should trigger two slots.
DOUBLE_PROMOTION_FEN = "4k3/P6P/8/8/8/8/8/4K3 w - - 0 1"


def _fen_to_tensor(fen: str) -> torch.Tensor:
    return torch.from_numpy(fen_to_simple_18(fen)).unsqueeze(0)


def _small_config(**overrides) -> dict:
    base = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 24,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "max_promotion_pawns": 2,
        "pawn_embed_dim": 8,
        "promotion_embed_dim": 4,
        "attn_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "gate_init": -2.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present():
    assert REGISTRY_KEY in available_models()


def test_promotion_type_constants_match_prototype():
    assert PROMOTION_TYPE_NAMES == ("Q", "R", "B", "N")
    assert PROMOTION_TYPE_COUNT == 4


def test_find_near_promotion_slots_white():
    board = _fen_to_tensor(WHITE_NEAR_PROMOTION_FEN)
    slots = find_near_promotion_slots(board, max_pawns=4)
    assert slots.valid.shape == (1, 4)
    assert slots.valid[0, 0].item() is True
    assert slots.valid[0, 1:].any().item() is False
    # White pawn on rank 7 = plane row 1, file 0 -> source_square = 1*8 + 0 = 8.
    assert int(slots.source_square[0, 0].item()) == 8
    # Promotion square = plane row 0, file 0 -> 0.
    assert int(slots.promote_square[0, 0].item()) == 0
    # own_color == 0 means white.
    assert int(slots.own_color[0, 0].item()) == 0
    # own_pawn_plane == 0 (white pawn).
    assert int(slots.own_pawn_plane[0, 0].item()) == 0


def test_find_near_promotion_slots_black():
    board = _fen_to_tensor(BLACK_NEAR_PROMOTION_FEN)
    slots = find_near_promotion_slots(board, max_pawns=4)
    assert slots.valid[0, 0].item() is True
    # Black pawn on rank 2 = plane row 6, file 0 -> source_square = 6*8 + 0 = 48.
    assert int(slots.source_square[0, 0].item()) == 48
    # Promotion square = plane row 7, file 0 -> 56.
    assert int(slots.promote_square[0, 0].item()) == 56
    # own_color == 1 means black.
    assert int(slots.own_color[0, 0].item()) == 1
    # own_pawn_plane == 6 (black pawn).
    assert int(slots.own_pawn_plane[0, 0].item()) == 6


def test_find_near_promotion_slots_initial_position_is_empty():
    board = _fen_to_tensor(INITIAL_FEN)
    slots = find_near_promotion_slots(board, max_pawns=4)
    assert slots.valid.any().item() is False


def test_find_near_promotion_slots_double_promotion_keeps_file_order():
    board = _fen_to_tensor(DOUBLE_PROMOTION_FEN)
    slots = find_near_promotion_slots(board, max_pawns=4)
    # Two slots valid (a7 and h7).
    assert int(slots.valid[0].sum().item()) == 2
    # File 0 (a-file) should come first by the lowest-file tie-break.
    src_squares = sorted(int(s.item()) for s in slots.source_square[0, :2])
    assert src_squares == [8, 15]  # rank 7 (plane row 1), files 0 and 7


def test_counterfactual_board_substitutes_promotion_piece_for_pawn():
    board = _fen_to_tensor(WHITE_NEAR_PROMOTION_FEN)
    model = build_model(REGISTRY_KEY, _small_config())
    model.eval()
    slots = find_near_promotion_slots(board, max_pawns=2)
    cf = build_promotion_counterfactuals(board, slots, model.promote_plane_lookup)
    assert cf.shape == (1, 2, 4, 18, 8, 8)
    # Inspect the first slot (a7 promotion).
    cf_slot = cf[0, 0]  # (4, 18, 8, 8)
    # In the counterfactual, the white pawn at plane 0, rank-idx 1, file 0
    # must be cleared in all four substituted boards.
    assert cf_slot[:, 0, 1, 0].abs().sum().item() == 0.0
    # The promoted piece must appear on plane row 0, file 0 of the right plane.
    # PROMOTION_TYPE_NAMES = ("Q", "R", "B", "N")
    # White Q is plane 4, R=3, B=2, N=1.
    expected_planes = (4, 3, 2, 1)
    for t, plane in enumerate(expected_planes):
        # The promoted plane should have exactly one 1.0 at promote_square.
        assert cf_slot[t, plane, 0, 0].item() == pytest.approx(1.0)
        # No other piece plane should fire at the promotion square.
        other_planes = [p for p in range(12) if p != plane]
        assert cf_slot[t, other_planes, 0, 0].abs().sum().item() == 0.0


def test_counterfactual_invalid_slot_leaves_board_unchanged():
    board = _fen_to_tensor(WHITE_NEAR_PROMOTION_FEN)
    model = build_model(REGISTRY_KEY, _small_config(max_promotion_pawns=4))
    slots = find_near_promotion_slots(board, max_pawns=4)
    cf = build_promotion_counterfactuals(board, slots, model.promote_plane_lookup)
    # Slot 0 is valid (a7 pawn), slots 1..3 are invalid in this position.
    for invalid_slot in (1, 2, 3):
        for promotion_type in range(4):
            assert torch.allclose(cf[0, invalid_slot, promotion_type], board[0])


def test_counterfactual_capture_promotion_clears_target_piece():
    # White pawn on a7, enemy black knight on a8 (capture-promotion target).
    fen = "n3k3/P7/8/8/8/8/8/4K3 w - - 0 1"
    board = _fen_to_tensor(fen)
    model = build_model(REGISTRY_KEY, _small_config())
    slots = find_near_promotion_slots(board, max_pawns=2)
    cf = build_promotion_counterfactuals(board, slots, model.promote_plane_lookup)
    # The black knight at plane 7 (black knight), rank-idx 0, file 0 must be
    # cleared in every substituted board for the a7 slot.
    cf_slot = cf[0, 0]
    assert cf_slot[:, 7, 0, 0].abs().sum().item() == 0.0
    # And the promotion piece must be exactly one piece on plane row 0 file 0.
    piece_total_at_promote_square = cf_slot[:, :12, 0, 0].sum(dim=1)
    assert torch.allclose(piece_total_at_promote_square, torch.ones(4))


def test_zero_overhead_gate_when_no_near_promotion_pawn():
    """Per the PFCT spec, gate must be exactly 0 on positions with no
    own near-promotion pawn (structural multiplication by has_promotion_pawn)."""
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config())
    model.eval()
    board = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(board)
    assert float(out["primitive_gate"].item()) == 0.0
    assert float(out["promotion_pawn_count"].item()) == 0.0
    assert float(out["primitive_logit_contribution"].item()) == 0.0
    # The final logit must equal base_logit when the gate is zero.
    assert torch.allclose(out["logits"], out["base_logit"], atol=1.0e-6)


def test_dominant_promotion_type_marked_neg_one_without_pawn():
    """The dominant-type diagnostic should be -1 when there is no
    near-promotion pawn so downstream slice reports can skip those rows."""
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config())
    model.eval()
    board = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out = model(board)
    assert float(out["promotion_dominant_type"].item()) == -1.0


def test_full_diagnostics_keys_are_present():
    model = build_model(REGISTRY_KEY, _small_config())
    model.eval()
    board = _fen_to_tensor(WHITE_NEAR_PROMOTION_FEN)
    with torch.no_grad():
        out = model(board)
    expected = {
        "logits",
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "primitive_gate_entropy",
        "primitive_logit_contribution",
        "promotion_pawn_count",
        "promotion_has_pawn",
        "promotion_attention_entropy",
        "promotion_dominant_type",
        "promotion_fanout_dispersion",
        "promotion_pawn_delta_max",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    }
    missing = expected - set(out.keys())
    assert not missing, f"Missing diagnostics keys: {sorted(missing)}"


def test_copy_baseline_fanout_ablation_zeroes_fanout_dispersion():
    """The copy_baseline_fanout ablation replaces the fanout with 4 copies of
    the baseline feature, so the fanout dispersion (around the mean) must be
    exactly zero on any sample with at least one near-promotion pawn."""
    model = build_model(REGISTRY_KEY, _small_config(ablation="copy_baseline_fanout"))
    model.eval()
    board = _fen_to_tensor(WHITE_NEAR_PROMOTION_FEN)
    with torch.no_grad():
        out = model(board)
    assert float(out["promotion_fanout_dispersion"].item()) == pytest.approx(0.0, abs=1.0e-6)


def test_zero_delta_ablation_collapses_to_base_logit():
    model = build_model(REGISTRY_KEY, _small_config(ablation="zero_delta"))
    model.eval()
    board = _fen_to_tensor(WHITE_NEAR_PROMOTION_FEN)
    with torch.no_grad():
        out = model(board)
    assert torch.allclose(out["logits"], out["base_logit"], atol=1.0e-6)
    assert float(out["primitive_delta"].item()) == 0.0


def test_force_open_gate_ablation_keeps_gate_at_one_when_pawn_present():
    model = build_model(REGISTRY_KEY, _small_config(ablation="force_open_gate"))
    model.eval()
    board_with_pawn = _fen_to_tensor(WHITE_NEAR_PROMOTION_FEN)
    board_no_pawn = _fen_to_tensor(INITIAL_FEN)
    with torch.no_grad():
        out_with = model(board_with_pawn)
        out_no = model(board_no_pawn)
    assert float(out_with["primitive_gate"].item()) == pytest.approx(1.0)
    assert float(out_no["primitive_gate"].item()) == pytest.approx(0.0)


def test_uniform_attention_ablation_has_max_entropy():
    """The uniform_attention ablation forces a 1/4-uniform distribution,
    so the per-pawn attention entropy (normalised) must be 1.0."""
    model = build_model(REGISTRY_KEY, _small_config(ablation="uniform_attention"))
    model.eval()
    board = _fen_to_tensor(WHITE_NEAR_PROMOTION_FEN)
    with torch.no_grad():
        out = model(board)
    assert float(out["promotion_attention_entropy"].item()) == pytest.approx(1.0, abs=1.0e-5)


def test_forward_shape_for_batched_mixed_input():
    model = build_model(REGISTRY_KEY, _small_config())
    model.eval()
    fens = [
        WHITE_NEAR_PROMOTION_FEN,
        INITIAL_FEN,
        DOUBLE_PROMOTION_FEN,
        BLACK_NEAR_PROMOTION_FEN,
    ]
    batch = torch.cat([_fen_to_tensor(fen) for fen in fens], dim=0)
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (4,)
    assert out["promotion_pawn_count"].shape == (4,)
    # The initial position must have zero pawn count; the others must be > 0.
    assert float(out["promotion_pawn_count"][0].item()) == 1.0
    assert float(out["promotion_pawn_count"][1].item()) == 0.0
    assert float(out["promotion_pawn_count"][2].item()) == 2.0
    assert float(out["promotion_pawn_count"][3].item()) == 1.0


def test_backward_pass_routes_gradients_through_head_and_trunk():
    torch.manual_seed(0)
    model = build_model(REGISTRY_KEY, _small_config()).train()
    fens = [WHITE_NEAR_PROMOTION_FEN, INITIAL_FEN]
    batch = torch.cat([_fen_to_tensor(fen) for fen in fens], dim=0)
    target = torch.tensor([1.0, 0.0])
    out = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], target)
    loss.backward()
    head_params = [
        p
        for name, p in model.named_parameters()
        if not name.startswith("trunk.") and p.requires_grad
    ]
    trunk_params = [
        p
        for name, p in model.trunk.named_parameters()
        if p.requires_grad
    ]
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in head_params
    ), "PFCT head received no gradient"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in trunk_params
    ), "i193 trunk received no gradient"


def test_config_yaml_passes_static_validation():
    config_path = IDEA_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["idea_id"] == "i246"
    assert config["mode"] == "puzzle_binary"
    assert config["device"] == "nvidia"
    assert config["data"]["encoding"] == "simple_18"
    assert config["model"]["name"] == REGISTRY_KEY
    assert config["model"]["num_classes"] == 1
    assert config["model"]["input_channels"] == 18


def test_allowed_ablations_match_documented_set():
    expected = {
        "none",
        "copy_baseline_fanout",
        "uniform_attention",
        "zero_delta",
        "force_open_gate",
        "trunk_only",
    }
    assert set(ALLOWED_ABLATIONS) == expected


def test_rejects_non_simple_18_input_channels():
    with pytest.raises(ValueError):
        PromotionAwareHead(input_channels=20)


def test_rejects_multi_class_num_classes():
    with pytest.raises(ValueError):
        PromotionAwareHead(num_classes=3)


def test_rejects_invalid_max_pawns():
    with pytest.raises(ValueError):
        PromotionAwareHead(max_promotion_pawns=0)
    with pytest.raises(ValueError):
        PromotionAwareHead(max_promotion_pawns=9)


def test_build_from_config_round_trip():
    cfg = _small_config()
    model = build_promotion_aware_head_from_config(cfg)
    assert isinstance(model, PromotionAwareHead)
    assert model.max_promotion_pawns == cfg["max_promotion_pawns"]
    assert model.ablation == "none"
