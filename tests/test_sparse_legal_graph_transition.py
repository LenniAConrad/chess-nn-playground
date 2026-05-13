"""Focused tests for the p035 Sparse Legal-Move Graph Transition primitive (SLMGT)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.sparse_legal_graph_transition import (
    ALLOWED_ABLATIONS,
    SparseLegalGraphTransition,
    build_sparse_legal_graph_transition_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "sparse_legal_graph_transition"
IDEA_DIR = Path("ideas/registry/p035_sparse_legal_graph_transition")
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
        "feature_dim": 8,
        "edge_hidden_dim": 8,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, SparseLegalGraphTransition)
    aliased = build_sparse_legal_graph_transition_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "feature_dim": 6,
            "edge_hidden_dim": 8,
            "head_hidden_dim": 16,
        }
    )
    assert aliased.trunk.channels == 12


def test_forward_shape_and_keys() -> None:
    model = SparseLegalGraphTransition(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "slmgt_degree_mean",
        "slmgt_edge_norm",
        "slmgt_edge_max",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = SparseLegalGraphTransition(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert model.w_self.weight.grad is not None
    assert model.w_neighbor.weight.grad is not None
    assert model.w_interact.weight.grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = SparseLegalGraphTransition(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_separable_phi_differs_from_full() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = SparseLegalGraphTransition(**cfg).eval()
    torch.manual_seed(0)
    separable = SparseLegalGraphTransition(**cfg, ablation="separable_phi").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        separable_out = separable(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], separable_out["primitive_delta_raw"]
    )


def test_uniform_adjacency_differs_from_full() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = SparseLegalGraphTransition(**cfg).eval()
    torch.manual_seed(0)
    dense = SparseLegalGraphTransition(**cfg, ablation="uniform_adjacency").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        dense_out = dense(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], dense_out["primitive_delta_raw"]
    )


def test_disable_gate_pins_gate_at_one() -> None:
    model = SparseLegalGraphTransition(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_shuffle_adjacency_remains_finite() -> None:
    torch.manual_seed(0)
    model = SparseLegalGraphTransition(**_toy_kwargs(), ablation="shuffle_adjacency").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        SparseLegalGraphTransition(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        SparseLegalGraphTransition(input_channels=12, num_classes=1)


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = SparseLegalGraphTransition(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p035"
    assert data["slug"] == "sparse_legal_graph_transition"
    assert data["implementation_kind"] == "bespoke_model"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p035"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
