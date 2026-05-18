"""Focused tests for the i257 Promotion Mate Slice Specialist."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.ideas.implementation import validate_idea_scaffold
from chess_nn_playground.ideas.implementation_kind import (
    analyze_model_wiring,
    detect_idea_implementation_kind,
)
from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.trunk.promotion_mate_slice_specialist import (
    PromotionMateSliceSpecialist,
    build_promotion_mate_slice_specialist_from_config,
)


FOLDER = Path("ideas/registry/i257_promotion_mate_slice_specialist")


def _toy_kwargs() -> dict[str, object]:
    return {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 16,
        "hidden_dim": 24,
        "depth": 1,
        "head_hidden_dim": 16,
        "type_embed_dim": 8,
        "dropout": 0.0,
        "use_batchnorm": False,
        "delta_bound": 1.5,
        "joint_delta_bound": 0.75,
        "max_promotion_candidates": 3,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "promotion_mate_slice_specialist" in available_models()
    model = build_model("promotion_mate_slice_specialist", _toy_kwargs())
    assert isinstance(model, PromotionMateSliceSpecialist)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = PromotionMateSliceSpecialist(**_toy_kwargs()).eval()
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",  # white pawn on 7th, near promotion
            "8/8/8/8/8/8/7p/K6k b - - 0 1",  # black pawn on 2nd, near promotion
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert isinstance(out, dict)
    expected = {
        "logits",
        "base_logit",
        "promotion_delta",
        "underpromotion_delta",
        "mate_delta",
        "joint_delta",
        "promotion_gate",
        "underpromotion_gate",
        "mate_gate",
        "joint_gate",
        "promotion_candidate_count",
        "promotion_best_type",
        "promotion_type_entropy",
        "underpromotion_margin",
        "mate_witness_count",
        "escape_square_count",
        "checking_move_count",
        "king_pressure",
        "mating_special_count",
        "mechanism_energy",
    }
    assert expected.issubset(out.keys())
    for key in expected:
        assert out[key].shape == (4,), key
        assert torch.isfinite(out[key]).all(), key


def test_bounded_delta_identity() -> None:
    """`|final - base| <= sum_k Delta_k` by construction."""

    cfg = dict(_toy_kwargs())
    model = PromotionMateSliceSpecialist(**cfg).eval()
    bound = 3 * cfg["delta_bound"] + cfg["joint_delta_bound"]
    torch.manual_seed(0)
    boards = torch.randn(16, 18, 8, 8)
    boards[:, 12] = (torch.rand(16, 1, 1) > 0.5).float()
    boards[:, :12] = (boards[:, :12] > 0.5).float()
    with torch.inference_mode():
        out = model(boards)
    diff = (out["logits"] - out["base_logit"]).abs()
    assert torch.all(diff <= bound + 1e-5), diff.max().item()


def test_promotion_candidate_count_detects_seventh_rank_pawn() -> None:
    cfg = dict(_toy_kwargs())
    model = PromotionMateSliceSpecialist(**cfg).eval()
    boards = _board_batch(
        [
            # White pawn on a7 (one push from promotion); side-to-move = white.
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",
            # No near-promotion pawns; side-to-move = white.
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # Black pawn on h2 (one push from promotion); side-to-move = black.
            "8/8/8/8/8/8/7p/K6k b - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    counts = out["promotion_candidate_count"]
    assert counts[0].item() >= 1.0
    assert counts[1].item() == 0.0
    assert counts[2].item() >= 1.0


def test_force_zero_gate_matches_base_logit() -> None:
    cfg = dict(_toy_kwargs())
    cfg["ablation"] = "force_zero_gate"
    model = PromotionMateSliceSpecialist(**cfg).eval()
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert torch.allclose(out["logits"], out["base_logit"], atol=1e-6)
    assert torch.allclose(out["promotion_gate"], torch.zeros(2))
    assert torch.allclose(out["mate_gate"], torch.zeros(2))


def test_trunk_only_ablation_zeroes_specialist_deltas() -> None:
    cfg = dict(_toy_kwargs())
    cfg["ablation"] = "trunk_only"
    model = PromotionMateSliceSpecialist(**cfg).eval()
    boards = _board_batch(
        [
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert torch.allclose(out["promotion_delta"], torch.zeros(2))
    assert torch.allclose(out["underpromotion_delta"], torch.zeros(2))
    assert torch.allclose(out["mate_delta"], torch.zeros(2))
    assert torch.allclose(out["joint_delta"], torch.zeros(2))
    assert torch.allclose(out["logits"], out["base_logit"], atol=1e-6)


def test_backward_gradients_reach_trunk_and_branches() -> None:
    model = PromotionMateSliceSpecialist(**_toy_kwargs())
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
        ]
    )
    out = model(boards)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out["logits"], torch.tensor([1.0, 0.0, 1.0])
    )
    loss.backward()

    trunk_param = next(model.encoder.parameters())
    cand_param = next(model.candidate_proj.parameters())
    base_param = next(model.base_head.parameters())
    type_param = next(model.type_descriptor_head.parameters())
    promo_param = next(model.promotion_summary_head.parameters())
    under_param = next(model.underpromotion_summary_head.parameters())
    mate_param = next(model.mate_summary_head.parameters())
    joint_param = next(model.joint_summary_head.parameters())

    for param in (
        trunk_param,
        cand_param,
        base_param,
        type_param,
        promo_param,
        under_param,
        mate_param,
        joint_param,
    ):
        assert param.grad is not None
        assert param.grad.abs().sum().item() > 0


@pytest.mark.parametrize(
    "ablation", list(PromotionMateSliceSpecialist.ALLOWED_ABLATIONS)
)
def test_all_ablations_produce_finite_logits(ablation: str) -> None:
    cfg = dict(_toy_kwargs())
    cfg["ablation"] = ablation
    model = PromotionMateSliceSpecialist(**cfg).eval()
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert out["logits"].shape == (3,)
    assert torch.isfinite(out["logits"]).all()


def test_structural_mask_zeroes_promotion_gate_without_candidates() -> None:
    """When there are no near-promotion pawns, the promotion gate must be 0."""

    cfg = dict(_toy_kwargs())
    model = PromotionMateSliceSpecialist(**cfg).eval()
    boards = _board_batch(
        [
            # Starting position: no pawns on the 7th rank for white.
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert out["promotion_candidate_count"].item() == 0.0
    assert out["promotion_gate"].item() == 0.0
    assert out["underpromotion_gate"].item() == 0.0
    assert out["joint_gate"].item() == 0.0


def test_idea_folder_validates_scaffold() -> None:
    report = validate_idea_scaffold(FOLDER)
    assert report["valid"], report


def test_idea_implementation_kind_is_bespoke() -> None:
    row = detect_idea_implementation_kind(FOLDER)
    assert row.detected_kind == "bespoke_model"
    assert row.implementation_status == "implemented"
    assert not row.issues, row.issues


def test_idea_model_py_does_not_use_research_packet_probe() -> None:
    model_py = (FOLDER / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called


def test_idea_model_py_round_trip_against_config() -> None:
    config = yaml.safe_load((FOLDER / "config.yaml").read_text(encoding="utf-8"))
    builder = build_promotion_mate_slice_specialist_from_config(config["model"])
    assert isinstance(builder, PromotionMateSliceSpecialist)
    assert not isinstance(builder, ResearchPacketProbe)
