"""Focused tests for the p042 Truncated Multiset Polynomial Pool primitive (TMPP)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.truncated_multiset_polynomial_pool import (
    ALLOWED_ABLATIONS,
    TruncatedMultisetPolynomialPool,
    build_truncated_multiset_polynomial_pool_from_config,
    truncated_elementary_symmetric_scan,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "truncated_multiset_polynomial_pool"
IDEA_DIR = Path("ideas/registry/p042_truncated_multiset_polynomial_pool")
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
        "latent_dim": 8,
        "degree": 3,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "coeff_norm": False,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, TruncatedMultisetPolynomialPool)
    aliased = build_truncated_multiset_polynomial_pool_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "latent_dim": 6,
            "degree": 2,
            "head_hidden_dim": 16,
        }
    )
    assert aliased.trunk.channels == 12
    assert aliased.degree == 2


def test_forward_shape_and_keys() -> None:
    model = TruncatedMultisetPolynomialPool(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "tmpp_active_mean",
        "tmpp_coeff_norm",
        "tmpp_coeff_e1",
        "tmpp_coeff_e2",
        "tmpp_coeff_e3",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = TruncatedMultisetPolynomialPool(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.token_proj.parameters()).grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = TruncatedMultisetPolynomialPool(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_first_order_only_differs_from_full_K() -> None:
    cfg = _toy_kwargs(degree=3)
    torch.manual_seed(0)
    full = TruncatedMultisetPolynomialPool(**cfg).eval()
    torch.manual_seed(0)
    first = TruncatedMultisetPolynomialPool(**cfg, ablation="first_order_only").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        first_out = first(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], first_out["primitive_delta_raw"]
    )


def test_uniform_mask_differs_from_occupancy_mask() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = TruncatedMultisetPolynomialPool(**cfg).eval()
    torch.manual_seed(0)
    dense = TruncatedMultisetPolynomialPool(**cfg, ablation="uniform_mask").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        dense_out = dense(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], dense_out["primitive_delta_raw"]
    )


def test_disable_gate_pins_gate_at_one() -> None:
    model = TruncatedMultisetPolynomialPool(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_polynomial_scan_matches_explicit_enumeration_k2() -> None:
    """Verify the elementary-symmetric scan against an explicit sum over pairs."""
    torch.manual_seed(0)
    u = torch.randn(2, 5, 3)
    mask = torch.tensor([[1, 1, 0, 1, 1], [0, 1, 1, 1, 0]], dtype=torch.float32)
    scan_out = truncated_elementary_symmetric_scan(u, mask, degree=2)
    # Reference e_1 = sum_i m_i u_i
    e1_ref = (u * mask.unsqueeze(-1)).sum(dim=1)
    # Reference e_2 = sum_{i<j} m_i m_j u_i u_j
    weighted = u * mask.unsqueeze(-1)
    n = weighted.shape[1]
    e2_ref = torch.zeros_like(e1_ref)
    for i in range(n):
        for j in range(i + 1, n):
            e2_ref = e2_ref + weighted[:, i] * weighted[:, j]
    assert torch.allclose(scan_out[:, 0], e1_ref, atol=1e-5)
    assert torch.allclose(scan_out[:, 1], e2_ref, atol=1e-5)


def test_polynomial_scan_token_order_invariance() -> None:
    """The coefficients are invariant to permutations of the active tokens."""
    torch.manual_seed(1)
    u = torch.randn(2, 6, 4)
    mask = torch.ones(2, 6)
    base = truncated_elementary_symmetric_scan(u, mask, degree=3)
    perm = torch.randperm(6)
    permuted = truncated_elementary_symmetric_scan(u[:, perm], mask[:, perm], degree=3)
    assert torch.allclose(base, permuted, atol=1e-5)


def test_shuffle_tokens_preserves_delta() -> None:
    """Token-shuffle ablation should not change the elementary-symmetric coefficients.

    LayerNorm + delta head reductions are symmetric, so the gated delta
    must match the unablated forward exactly.
    """
    torch.manual_seed(0)
    cfg = _toy_kwargs(degree=2, coeff_norm=False)
    full = TruncatedMultisetPolynomialPool(**cfg).eval()
    torch.manual_seed(0)
    shuffled = TruncatedMultisetPolynomialPool(**cfg, ablation="shuffle_tokens").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        shuffled_out = shuffled(boards)
    assert torch.allclose(
        full_out["primitive_delta_raw"], shuffled_out["primitive_delta_raw"], atol=1e-5
    )


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        TruncatedMultisetPolynomialPool(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        TruncatedMultisetPolynomialPool(input_channels=12, num_classes=1)


def test_rejects_invalid_degree() -> None:
    with pytest.raises(ValueError):
        TruncatedMultisetPolynomialPool(**_toy_kwargs(degree=0))
    with pytest.raises(ValueError):
        TruncatedMultisetPolynomialPool(**_toy_kwargs(degree=7))


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = TruncatedMultisetPolynomialPool(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p042"
    assert data["slug"] == "truncated_multiset_polynomial_pool"
    assert data["implementation_kind"] == "bespoke_model"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p042"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
