"""Focused tests for the p043 Grassmann Rook-Matching Pool primitive (GRMP)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.grassmann_rook_pool import (
    ALLOWED_ABLATIONS,
    GrassmannRookPool,
    build_grassmann_rook_pool_from_config,
    grassmann_rook_matching_coefficients,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "grassmann_rook_pool"
IDEA_DIR = Path("ideas/registry/p043_grassmann_rook_pool")
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
        "num_attackers": 4,
        "num_defenders": 4,
        "token_dim": 12,
        "score_channels": 4,
        "degree": 2,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, GrassmannRookPool)
    aliased = build_grassmann_rook_pool_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "num_attackers": 4,
            "num_defenders": 4,
            "token_dim": 8,
            "score_channels": 4,
            "degree": 1,
            "head_hidden_dim": 16,
        }
    )
    assert aliased.trunk.channels == 12
    assert aliased.degree == 1


def test_forward_shape_and_keys() -> None:
    model = GrassmannRookPool(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "grmp_attacker_count",
        "grmp_defender_count",
        "grmp_coeff_norm",
        "grmp_coeff_e1",
        "grmp_coeff_e2",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = GrassmannRookPool(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.attacker_pool.parameters()).grad is not None
    assert next(model.defender_pool.parameters()).grad is not None
    assert next(model.bilinear.parameters()).grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = GrassmannRookPool(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_drop_exclusion_differs_from_full() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = GrassmannRookPool(**cfg).eval()
    torch.manual_seed(0)
    no_excl = GrassmannRookPool(**cfg, ablation="drop_exclusion").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        no_out = no_excl(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], no_out["primitive_delta_raw"]
    )


def test_scalar_score_differs_from_multichannel() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = GrassmannRookPool(**cfg).eval()
    torch.manual_seed(0)
    scalar = GrassmannRookPool(**cfg, ablation="scalar_score").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        scalar_out = scalar(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], scalar_out["primitive_delta_raw"]
    )


def test_disable_gate_pins_gate_at_one() -> None:
    model = GrassmannRookPool(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_matching_coefficient_e1_matches_sum_of_active_edges() -> None:
    """e_1 of the matching polynomial is just the sum of active edges."""
    torch.manual_seed(2)
    z = torch.randn(3, 3, 3, 2)
    am = torch.tensor([[1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
    dm = torch.tensor([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=torch.float32)
    coeff = grassmann_rook_matching_coefficients(z, am, dm, degree=1)
    expected = (
        z * am.unsqueeze(-1).unsqueeze(-1) * dm.unsqueeze(1).unsqueeze(-1)
    ).sum(dim=(1, 2))
    assert torch.allclose(coeff[:, 0], expected, atol=1e-5)


def test_matching_coefficient_e2_matches_explicit_disjoint_pairs() -> None:
    """e_2 should equal the explicit sum over row/column-disjoint edge pairs."""
    torch.manual_seed(3)
    z = torch.randn(2, 3, 3, 1)
    am = torch.ones(2, 3)
    dm = torch.ones(2, 3)
    coeff = grassmann_rook_matching_coefficients(z, am, dm, degree=2)
    weighted = z * am.unsqueeze(-1).unsqueeze(-1) * dm.unsqueeze(1).unsqueeze(-1)
    batch = z.shape[0]
    channels = z.shape[-1]
    explicit = torch.zeros(batch, channels)
    for i1 in range(3):
        for j1 in range(3):
            for i2 in range(3):
                for j2 in range(3):
                    if i1 == i2 or j1 == j2:
                        continue
                    if (i1, j1) >= (i2, j2):
                        continue
                    explicit = explicit + weighted[:, i1, j1] * weighted[:, i2, j2]
    assert torch.allclose(coeff[:, 1], explicit, atol=1e-4)


def test_matching_coefficient_drop_exclusion_differs() -> None:
    """The non-exclusion variant should produce different e_2 than the rook variant."""
    torch.manual_seed(4)
    z = torch.randn(2, 4, 4, 1)
    am = torch.ones(2, 4)
    dm = torch.ones(2, 4)
    rook = grassmann_rook_matching_coefficients(z, am, dm, degree=2, exclude_rows_cols=True)
    no_excl = grassmann_rook_matching_coefficients(z, am, dm, degree=2, exclude_rows_cols=False)
    assert not torch.allclose(rook[:, 1], no_excl[:, 1])


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        GrassmannRookPool(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        GrassmannRookPool(input_channels=12, num_classes=1)


def test_rejects_invalid_degree() -> None:
    with pytest.raises(ValueError):
        GrassmannRookPool(**_toy_kwargs(degree=0))
    with pytest.raises(ValueError):
        GrassmannRookPool(**_toy_kwargs(degree=4))


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = GrassmannRookPool(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_degree_3_anchor_scan_runs_and_returns_finite() -> None:
    torch.manual_seed(0)
    model = GrassmannRookPool(**_toy_kwargs(degree=3)).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()
    assert "grmp_coeff_e3" in out


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p043"
    assert data["slug"] == "grassmann_rook_pool"
    assert data["implementation_kind"] == "bespoke_model"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p043"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
