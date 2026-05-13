"""Focused tests for the p044 Weighted Hodge Projector primitive (WHP)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.weighted_hodge_projector import (
    ALLOWED_ABLATIONS,
    WeightedHodgeProjector,
    _build_incidence_matrices,
    build_weighted_hodge_projector_from_config,
    hodge_decompose,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "weighted_hodge_projector"
IDEA_DIR = Path("ideas/registry/p044_weighted_hodge_projector")
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
        "flow_channels": 3,
        "edge_feature_dim": 8,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "solve_eps": 1.0e-2,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, WeightedHodgeProjector)
    aliased = build_weighted_hodge_projector_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "flow_channels": 2,
            "edge_feature_dim": 8,
        }
    )
    assert aliased.trunk.channels == 12
    assert aliased.flow_channels == 2


def test_incidence_orthogonality_d1t_d0t_zero() -> None:
    """The grid complex is well-formed: D_1^T D_0^T should be zero."""
    d0, d1 = _build_incidence_matrices()
    # d0 is (V, E); d1 is (E, F). image(D_0^T) is the gradient subspace,
    # image(D_1) is the curl subspace. Their orthogonality (in the
    # unweighted inner product) is encoded as d0 @ d1 == 0.
    product = d0 @ d1
    assert torch.allclose(product, torch.zeros_like(product), atol=1e-6)


def test_hodge_decomposition_sums_to_flow() -> None:
    d0, d1 = _build_incidence_matrices()
    n_edges = d0.shape[1]
    torch.manual_seed(0)
    flow = torch.randn(2, n_edges, 3)
    weights = torch.rand(2, n_edges) + 0.5
    g, c, h = hodge_decompose(flow, weights, d0, d1, eps=1.0e-3)
    reconstructed = g + c + h
    assert torch.allclose(reconstructed, flow, atol=1e-4)


def test_hodge_decomposition_uniform_metric_orthogonality() -> None:
    """With W = I and eps -> 0, the three components are orthogonal in L2."""
    d0, d1 = _build_incidence_matrices()
    n_edges = d0.shape[1]
    torch.manual_seed(1)
    flow = torch.randn(2, n_edges, 2)
    weights = torch.ones(2, n_edges)
    g, c, h = hodge_decompose(flow, weights, d0, d1, eps=1.0e-6)
    # Up to the eps-shift, G and C are orthogonal.
    gc_inner = (g * c).sum(dim=1)
    assert gc_inner.abs().max() < 1e-2


def test_forward_shape_and_keys() -> None:
    model = WeightedHodgeProjector(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "whp_gradient_energy",
        "whp_curl_energy",
        "whp_harmonic_energy",
        "whp_flow_energy",
        "whp_weight_mean",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = WeightedHodgeProjector(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.edge_proj.parameters()).grad is not None
    assert model.flow_head.weight.grad is not None
    assert model.metric_head.weight.grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = WeightedHodgeProjector(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_uniform_metric_differs_from_learned_metric() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = WeightedHodgeProjector(**cfg).eval()
    torch.manual_seed(0)
    uniform = WeightedHodgeProjector(**cfg, ablation="uniform_metric").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        uniform_out = uniform(boards)
    # Mean weight should be exactly 1.0 under uniform_metric.
    assert torch.allclose(
        uniform_out["whp_weight_mean"],
        torch.ones_like(uniform_out["whp_weight_mean"]),
    )
    assert not torch.allclose(
        full_out["primitive_delta_raw"], uniform_out["primitive_delta_raw"]
    )


def test_drop_curl_zeros_curl_energy() -> None:
    model = WeightedHodgeProjector(**_toy_kwargs(), ablation="drop_curl").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["whp_curl_energy"], torch.zeros_like(out["whp_curl_energy"]))


def test_drop_gradient_zeros_gradient_energy() -> None:
    model = WeightedHodgeProjector(**_toy_kwargs(), ablation="drop_gradient").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(
        out["whp_gradient_energy"], torch.zeros_like(out["whp_gradient_energy"])
    )


def test_drop_harmonic_zeros_harmonic_energy() -> None:
    model = WeightedHodgeProjector(**_toy_kwargs(), ablation="drop_harmonic").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(
        out["whp_harmonic_energy"], torch.zeros_like(out["whp_harmonic_energy"])
    )


def test_disable_gate_pins_gate_at_one() -> None:
    model = WeightedHodgeProjector(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        WeightedHodgeProjector(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        WeightedHodgeProjector(input_channels=12, num_classes=1)


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = WeightedHodgeProjector(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p044"
    assert data["slug"] == "weighted_hodge_projector"
    assert data["implementation_kind"] == "bespoke_model"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p044"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
