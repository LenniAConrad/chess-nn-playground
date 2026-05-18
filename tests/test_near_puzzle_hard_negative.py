"""Focused tests for the p055 Near-Puzzle Hard-Negative Veto primitive (NPHN)."""
from __future__ import annotations

from pathlib import Path

import chess

import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.near_puzzle_hard_negative import (
    ALLOWED_ABLATIONS,
    NearPuzzleHardNegativePrimitive,
    build_near_puzzle_hard_negative_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "near_puzzle_hard_negative"
IDEA_DIR = Path("ideas/registry/p055_near_puzzle_hard_negative")
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
        "num_candidates": 4,
        "num_replies": 4,
        "token_dim": 8,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, NearPuzzleHardNegativePrimitive)
    aliased = build_near_puzzle_hard_negative_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "num_candidates": 4,
            "num_replies": 4,
            "token_dim": 8,
            "head_hidden_dim": 16,
        }
    )
    assert aliased.trunk.channels == 12
    assert aliased.num_candidates == 4


def test_forward_shape_and_keys() -> None:
    model = NearPuzzleHardNegativePrimitive(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_contribution",
        "nphn_veto_pressure",
        "nphn_forcedness_gap",
        "nphn_forcedness_at_mstar",
        "nphn_legality_discount",
        "nphn_candidate_concentration",
        "nphn_candidate_gap",
        "nphn_reply_availability",
        "nphn_reply_channel_information",
        "nphn_attack_defense_balance",
        "nphn_king_escape_pressure",
        "nphn_defender_overload_asymmetry",
        "nphn_counterpressure",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = NearPuzzleHardNegativePrimitive(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.candidate_pool.parameters()).grad is not None
    assert next(model.reply_pool.parameters()).grad is not None
    assert next(model.veto_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_primitive_delta_is_non_positive() -> None:
    """By construction the veto head can only lower the puzzle logit."""
    torch.manual_seed(0)
    model = NearPuzzleHardNegativePrimitive(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert (out["primitive_delta"] <= 1e-6).all()
    assert (out["primitive_contribution"] <= 1e-6).all()


def test_zero_delta_recovers_trunk_logit() -> None:
    model = NearPuzzleHardNegativePrimitive(
        **_toy_kwargs(), ablation="zero_delta"
    ).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_no_replies_zeroes_reply_features() -> None:
    """`no_replies` ablation must zero out Avail, ReplyMass, and RCI."""
    torch.manual_seed(0)
    model = NearPuzzleHardNegativePrimitive(
        **_toy_kwargs(), ablation="no_replies"
    ).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(
        out["nphn_reply_availability"], torch.zeros_like(out["nphn_reply_availability"])
    )
    assert torch.allclose(
        out["nphn_reply_channel_information"],
        torch.zeros_like(out["nphn_reply_channel_information"]),
    )


def test_no_legality_discount_zeroes_disc() -> None:
    torch.manual_seed(0)
    model = NearPuzzleHardNegativePrimitive(
        **_toy_kwargs(), ablation="no_legality_discount"
    ).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(
        out["nphn_legality_discount"], torch.zeros_like(out["nphn_legality_discount"])
    )


def test_concentration_only_zeroes_extra_features() -> None:
    torch.manual_seed(0)
    model = NearPuzzleHardNegativePrimitive(
        **_toy_kwargs(), ablation="concentration_only"
    ).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    for key in (
        "nphn_legality_discount",
        "nphn_reply_availability",
        "nphn_reply_channel_information",
        "nphn_king_escape_pressure",
        "nphn_defender_overload_asymmetry",
        "nphn_attack_defense_balance",
        "nphn_counterpressure",
        "nphn_forcedness_gap",
        "nphn_forcedness_at_mstar",
    ):
        assert torch.allclose(out[key], torch.zeros_like(out[key])), key


def test_disable_gate_pins_gate_at_one() -> None:
    model = NearPuzzleHardNegativePrimitive(
        **_toy_kwargs(), ablation="disable_gate"
    ).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_no_overload_zeroes_doa() -> None:
    torch.manual_seed(0)
    model = NearPuzzleHardNegativePrimitive(
        **_toy_kwargs(), ablation="no_overload"
    ).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(
        out["nphn_defender_overload_asymmetry"],
        torch.zeros_like(out["nphn_defender_overload_asymmetry"]),
    )


def test_no_king_escape_zeroes_kep() -> None:
    torch.manual_seed(0)
    model = NearPuzzleHardNegativePrimitive(
        **_toy_kwargs(), ablation="no_king_escape"
    ).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(
        out["nphn_king_escape_pressure"],
        torch.zeros_like(out["nphn_king_escape_pressure"]),
    )


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        NearPuzzleHardNegativePrimitive(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        NearPuzzleHardNegativePrimitive(input_channels=12, num_classes=1)


def test_rejects_invalid_token_counts() -> None:
    with pytest.raises(ValueError):
        NearPuzzleHardNegativePrimitive(**_toy_kwargs(num_candidates=1))
    with pytest.raises(ValueError):
        NearPuzzleHardNegativePrimitive(**_toy_kwargs(num_replies=1))


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = NearPuzzleHardNegativePrimitive(
            **_toy_kwargs(), ablation=ablation
        ).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p055"
    assert data["slug"] == "near_puzzle_hard_negative"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p055"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
