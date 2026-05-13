"""Focused tests for the p031 Legal-Move Laplacian Resolvent primitive (LM-LPP)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.legal_move_graph import (
    SQUARES,
    compute_legal_move_graph,
)
from chess_nn_playground.models.primitives.legal_move_laplacian_resolvent import (
    ALLOWED_ABLATIONS,
    LegalMoveLaplacianResolvent,
    build_legal_move_laplacian_resolvent_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "legal_move_laplacian_resolvent"
IDEA_DIR = Path("ideas/registry/p031_legal_move_laplacian_resolvent")

ROOK_MATE_FEN = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"
SCHOLAR_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3"


def _board_batch(fens: list[str]) -> torch.Tensor:
    arrays = [fen_to_simple_18(fen) for fen in fens]
    return torch.from_numpy(np.stack(arrays, axis=0)).float()


def _toy_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 32,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "feature_dim": 8,
        "neumann_terms": 3,
        "alpha_init": 0.25,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    cfg = _toy_kwargs()
    model = build_model(REGISTRY_KEY, cfg)
    assert isinstance(model, LegalMoveLaplacianResolvent)
    # Alias keys are accepted -- pass only the alias names to verify the
    # fallback path in the builder (no canonical ``trunk_*`` keys present).
    aliased_cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 12,             # aliased -> trunk_channels
        "hidden_dim": 24,           # aliased -> trunk_hidden_dim
        "depth": 1,                 # aliased -> trunk_depth
        "dropout": 0.0,             # aliased -> trunk_dropout
        "use_batchnorm": False,     # aliased -> trunk_use_batchnorm
        "feature_dim": 8,
        "K": 2,                     # aliased -> neumann_terms
        "head_hidden_dim": 16,
    }
    aliased = build_legal_move_laplacian_resolvent_from_config(aliased_cfg)
    assert isinstance(aliased, LegalMoveLaplacianResolvent)
    assert aliased.trunk.channels == 12
    assert aliased.neumann_terms == 2


def test_legal_move_graph_starting_position_has_expected_degree() -> None:
    boards = _board_batch([chess.STARTING_FEN])
    graph = compute_legal_move_graph(boards)
    assert graph.adjacency.shape == (1, SQUARES, SQUARES)
    # White-to-move; the 16 white pieces should generate at least one legal
    # destination on average. Knights have two moves each, pawns have two
    # pushes available -> expect total degree to be > 0.
    total_edges = graph.adjacency.sum().item()
    assert total_edges > 0
    # Symmetry sanity: from rank 1 (plane row 7) a white knight reaches two squares.
    # b1 -> a3 / c3. b1 is plane row 7, file 1 -> 57. a3 is plane row 5, file 0 -> 40.
    assert graph.adjacency[0, 57, 40].item() > 0.5


def test_legal_move_graph_drops_edges_into_own_pieces() -> None:
    boards = _board_batch([chess.STARTING_FEN])
    graph = compute_legal_move_graph(boards)
    # White rook on a1 (plane row 7, file 0 -> 56) cannot move into the
    # white pawn on a2 (plane row 6, file 0 -> 48).
    assert graph.adjacency[0, 56, 48].item() == 0.0


def test_forward_shape_and_keys() -> None:
    model = LegalMoveLaplacianResolvent(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "primitive_contribution",
        "lmlpp_alpha",
        "lmlpp_mean_feature_norm",
        "lmlpp_max_feature_norm",
        "lmlpp_degree_mean",
    ):
        assert key in out, key
        assert out[key].shape == (2,), key
    assert torch.all(out["primitive_gate"] >= 0.0)
    assert torch.all(out["primitive_gate"] <= 1.0)


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = LegalMoveLaplacianResolvent(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert model.alpha_logit.grad is not None
    assert model.theta.weight.grad is not None
    assert model.piece_edge_weights.grad is not None
    head_param = next(model.delta_head.parameters())
    gate_param = next(model.gate_head.parameters())
    assert head_param.grad is not None
    assert gate_param.grad is not None


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = LegalMoveLaplacianResolvent(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["primitive_contribution"], torch.zeros_like(out["primitive_contribution"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_matches_zero_delta() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    model_z = LegalMoveLaplacianResolvent(**cfg, ablation="zero_delta").eval()
    torch.manual_seed(0)
    model_t = LegalMoveLaplacianResolvent(**cfg, ablation="trunk_only").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out_z = model_z(boards)
        out_t = model_t(boards)
    assert torch.allclose(out_z["logits"], out_t["logits"])


def test_zero_alpha_yields_identity_propagation() -> None:
    cfg = _toy_kwargs()
    model = LegalMoveLaplacianResolvent(**cfg, ablation="zero_alpha").eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["lmlpp_alpha"], torch.zeros_like(out["lmlpp_alpha"]))
    # max_norm and mean_norm should still be finite (the seed features
    # propagate through Theta even with alpha=0).
    assert torch.isfinite(out["lmlpp_mean_feature_norm"]).all()


def test_k1_gat_rebrand_uses_single_hop() -> None:
    cfg = _toy_kwargs(neumann_terms=4)
    torch.manual_seed(0)
    full = LegalMoveLaplacianResolvent(**cfg).eval()
    torch.manual_seed(0)
    k1 = LegalMoveLaplacianResolvent(**cfg, ablation="k1_gat_rebrand").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out_full = full(boards)
        out_k1 = k1(boards)
    # The two models should disagree on at least one input -- the K=4
    # resolvent and the K=1 collapse must not match exactly when alpha is
    # initialised non-trivially.
    full.alpha_logit.data.fill_(1.0)
    k1.alpha_logit.data.fill_(1.0)
    with torch.no_grad():
        out_full = full(boards)
        out_k1 = k1(boards)
    assert not torch.allclose(out_full["primitive_delta_raw"], out_k1["primitive_delta_raw"])


def test_disable_gate_pins_gate_at_one() -> None:
    model = LegalMoveLaplacianResolvent(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_shuffle_adjacency_does_not_crash_and_keeps_finite_logits() -> None:
    torch.manual_seed(0)
    model = LegalMoveLaplacianResolvent(**_toy_kwargs(), ablation="shuffle_adjacency").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN, SCHOLAR_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    assert out["logits"].shape == (3,)


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        LegalMoveLaplacianResolvent(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        LegalMoveLaplacianResolvent(input_channels=12, num_classes=1)


def test_rejects_invalid_neumann_terms() -> None:
    with pytest.raises(ValueError):
        LegalMoveLaplacianResolvent(input_channels=18, num_classes=1, neumann_terms=0)
    with pytest.raises(ValueError):
        LegalMoveLaplacianResolvent(input_channels=18, num_classes=1, neumann_terms=9)


def test_idea_yaml_metadata() -> None:
    idea_file = IDEA_DIR / "idea.yaml"
    data = yaml.safe_load(idea_file.read_text(encoding="utf-8"))
    assert data["idea_id"] == "p031"
    assert data["slug"] == "legal_move_laplacian_resolvent"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] in {"implemented", "tested"}


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p031"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
    assert cfg["mode"] == "puzzle_binary"


def test_all_allowed_ablations_have_handlers() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = LegalMoveLaplacianResolvent(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation
