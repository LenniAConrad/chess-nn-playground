"""Focused tests for the i256 Near Puzzle Rejection Specialist."""

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
from chess_nn_playground.models.trunk.near_puzzle_rejection_specialist import (
    NearPuzzleRejectionSpecialist,
    build_near_puzzle_rejection_specialist_from_config,
)


FOLDER = Path("ideas/registry/i256_near_puzzle_rejection_specialist")


def _toy_kwargs() -> dict[str, object]:
    return {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 16,
        "hidden_dim": 24,
        "depth": 1,
        "per_square_hidden": 16,
        "head_hidden_dim": 16,
        "dropout": 0.0,
        "use_batchnorm": False,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "near_puzzle_rejection_specialist" in available_models()
    model = build_model("near_puzzle_rejection_specialist", _toy_kwargs())
    assert isinstance(model, NearPuzzleRejectionSpecialist)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = NearPuzzleRejectionSpecialist(**_toy_kwargs()).eval()
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert isinstance(out, dict)
    expected = {
        "logits",
        "raw_claim_logit",
        "reply_veto_logit",
        "max_forcedness_gap",
        "top2_forcedness_gap",
        "forcedness_gap_entropy",
        "effective_candidate_count",
        "selected_candidate_count",
        "defender_overload",
        "king_escape_pressure",
        "claim_mass",
        "reply_escape_mass",
        "own_piece_count",
        "mechanism_energy",
    }
    assert expected.issubset(out.keys())
    for key in expected:
        assert out[key].shape == (2,), key
        assert torch.isfinite(out[key]).all(), key


def test_rejection_identity_is_softplus_only_subtract() -> None:
    """`final_logit` must never exceed `raw_claim_logit` by construction."""

    model = NearPuzzleRejectionSpecialist(**_toy_kwargs()).eval()
    torch.manual_seed(0)
    boards = torch.randn(8, 18, 8, 8)
    boards[:, 12] = 1.0  # side-to-move plane
    boards[:, :12] = (boards[:, :12] > 0.5).float()
    with torch.inference_mode():
        out = model(boards)
    diff = out["logits"] - out["raw_claim_logit"]
    assert torch.all(diff <= 1e-5), diff


def test_backward_gradients_reach_trunk_and_heads() -> None:
    model = NearPuzzleRejectionSpecialist(**_toy_kwargs())
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
        ]
    )
    out = model(boards)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out["logits"], torch.tensor([1.0, 0.0])
    )
    loss.backward()

    trunk_param = next(model.encoder.parameters())
    claim_param = next(model.claim_head.parameters())
    reply_param = next(model.reply_head.parameters())
    overload_param = next(model.overload_score_head.parameters())
    king_param = next(model.king_escape_head.parameters())
    concentration_param = next(model.concentration_head.parameters())
    raw_claim_param = next(model.raw_claim_head.parameters())
    veto_param = next(model.veto_head.parameters())

    for param in (
        trunk_param,
        claim_param,
        reply_param,
        overload_param,
        king_param,
        concentration_param,
        raw_claim_param,
        veto_param,
    ):
        assert param.grad is not None
        assert param.grad.abs().sum().item() > 0


@pytest.mark.parametrize("ablation", list(NearPuzzleRejectionSpecialist.ALLOWED_ABLATIONS))
def test_all_ablations_produce_finite_logits(ablation: str) -> None:
    cfg = dict(_toy_kwargs())
    cfg["ablation"] = ablation
    model = NearPuzzleRejectionSpecialist(**cfg).eval()
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()


def test_trunk_only_ablation_zeroes_specialist_outputs() -> None:
    cfg = dict(_toy_kwargs())
    cfg["ablation"] = "trunk_only"
    model = NearPuzzleRejectionSpecialist(**cfg).eval()
    boards = _board_batch(
        ["6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"]
    )
    with torch.inference_mode():
        out = model(boards)
    assert torch.allclose(out["defender_overload"], torch.zeros(1))
    assert torch.allclose(out["king_escape_pressure"], torch.zeros(1))
    # final must equal raw_claim - softplus(0) = raw_claim - log(2)
    expected_final = out["raw_claim_logit"] - torch.nn.functional.softplus(
        out["reply_veto_logit"]
    )
    assert torch.allclose(out["logits"], expected_final, atol=1e-6)


def test_idea_folder_validates_scaffold() -> None:
    # Scaffold-level validation guards everything except the data parquet paths;
    # data parquets are environmental and absent from this checkout.
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
    builder = build_near_puzzle_rejection_specialist_from_config(config["model"])
    assert isinstance(builder, NearPuzzleRejectionSpecialist)
    assert not isinstance(builder, ResearchPacketProbe)
