"""Focused tests for the p048 Candidate Move Forcedness primitive (CMF)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.candidate_move_forcedness import (
    ALLOWED_ABLATIONS,
    EDGE_FEATURE_DIM,
    CandidateMoveForcedness,
    build_candidate_move_forcedness_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "candidate_move_forcedness"
IDEA_DIR = Path("ideas/registry/p048_candidate_move_forcedness")
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
        "token_dim": 16,
        "score_hidden_dim": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "topk": 4,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, CandidateMoveForcedness)
    aliased = build_candidate_move_forcedness_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "token_dim": 12,
            "score_hidden_dim": 12,
            "head_hidden_dim": 16,
            "topk": 2,
        }
    )
    assert aliased.trunk.channels == 12
    assert aliased.token_dim == 12
    assert aliased.topk == 2


def test_forward_shape_and_keys() -> None:
    model = CandidateMoveForcedness(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "cmf_top1_score",
        "cmf_gap12",
        "cmf_topk_mass",
        "cmf_entropy",
        "cmf_move_count",
        "cmf_check_peak",
        "cmf_capture_peak",
        "cmf_promotion_peak",
        "cmf_see_peak",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = CandidateMoveForcedness(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.token_proj.parameters()).grad is not None
    assert next(model.score_mlp.parameters()).grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None
    assert model.move_type_embed.weight.grad is not None
    assert model.direction_embed.weight.grad is not None


def test_legal_move_count_positive_on_starting_board() -> None:
    model = CandidateMoveForcedness(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    # ``compute_legal_move_graph`` emits pseudo-legal threat edges
    # (pawn captures into empty squares are kept for threat
    # geometry), so the count on the starting position is larger
    # than the 20 strictly-legal moves. We just assert the count is
    # in a sane bounded range so the helper is wired in correctly.
    move_count = out["cmf_move_count"].item()
    assert 20.0 <= move_count <= 60.0


def test_zero_delta_recovers_trunk_logit() -> None:
    model = CandidateMoveForcedness(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_matches_zero_delta() -> None:
    """``trunk_only`` is a semantic alias of ``zero_delta``."""
    torch.manual_seed(0)
    zero = CandidateMoveForcedness(**_toy_kwargs(), ablation="zero_delta").eval()
    torch.manual_seed(0)
    trunk_only = CandidateMoveForcedness(**_toy_kwargs(), ablation="trunk_only").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        zero_out = zero(boards)
        trunk_out = trunk_only(boards)
    assert torch.allclose(zero_out["logits"], trunk_out["logits"])


def test_disable_gate_pins_gate_at_one() -> None:
    model = CandidateMoveForcedness(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_deterministic_score_differs_from_full() -> None:
    """``deterministic_score`` skips the score MLP; deltas differ from `none`."""
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = CandidateMoveForcedness(**cfg).eval()
    torch.manual_seed(0)
    det = CandidateMoveForcedness(**cfg, ablation="deterministic_score").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        det_out = det(boards)
    assert not torch.allclose(
        full_out["cmf_top1_score"], det_out["cmf_top1_score"]
    )


def test_dense_edges_alters_move_count() -> None:
    cfg = _toy_kwargs()
    full = CandidateMoveForcedness(**cfg).eval()
    dense = CandidateMoveForcedness(**cfg, ablation="dense_edges").eval()
    boards = _board_batch([ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        dense_out = dense(boards)
    # dense_edges replaces the legal adjacency with (1 - I), so the move
    # count must jump to 64 * 63.
    assert dense_out["cmf_move_count"].item() == pytest.approx(64 * 63, abs=1.0)
    assert full_out["cmf_move_count"].item() < dense_out["cmf_move_count"].item()


def test_edge_feature_dim_is_fourteen() -> None:
    assert EDGE_FEATURE_DIM == 14
    assert CandidateMoveForcedness.EDGE_FEATURE_DIM == 14


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        CandidateMoveForcedness(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        CandidateMoveForcedness(input_channels=12, num_classes=1)


def test_rejects_invalid_token_dim() -> None:
    with pytest.raises(ValueError):
        CandidateMoveForcedness(**_toy_kwargs(token_dim=1))


def test_rejects_invalid_topk() -> None:
    with pytest.raises(ValueError):
        CandidateMoveForcedness(**_toy_kwargs(topk=0))


def test_rejects_invalid_softmax_temperature() -> None:
    with pytest.raises(ValueError):
        CandidateMoveForcedness(**_toy_kwargs(softmax_temperature=0.0))


def test_rejects_num_classes_other_than_one() -> None:
    with pytest.raises(ValueError):
        CandidateMoveForcedness(num_classes=2)


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = CandidateMoveForcedness(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p048"
    assert data["slug"] == "candidate_move_forcedness"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p048"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
