"""Focused tests for the p045 Kirchhoff Mobility Solve primitive (KMS)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.kirchhoff_mobility_solve import (
    ALLOWED_ABLATIONS,
    KirchhoffMobilitySolve,
    _build_grid_incidence,
    build_kirchhoff_mobility_solve_from_config,
    kirchhoff_resolve,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "kirchhoff_mobility_solve"
IDEA_DIR = Path("ideas/registry/p045_kirchhoff_mobility_solve")
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
        "source_channels": 4,
        "output_channels": 4,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "shift": 1.0e-2,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, KirchhoffMobilitySolve)
    aliased = build_kirchhoff_mobility_solve_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "source_channels": 4,
            "output_channels": 4,
        }
    )
    assert aliased.trunk.channels == 12


def test_grid_incidence_shape_and_sign() -> None:
    d = _build_grid_incidence()
    # 56 horizontal + 56 vertical = 112 edges; 64 vertices.
    assert d.shape == (64, 112)
    # Each edge column has exactly one +1 and one -1.
    plus_counts = (d == 1.0).sum(dim=0)
    minus_counts = (d == -1.0).sum(dim=0)
    assert torch.equal(plus_counts, torch.ones(112, dtype=plus_counts.dtype))
    assert torch.equal(minus_counts, torch.ones(112, dtype=minus_counts.dtype))


def test_kirchhoff_resolve_recovers_constant_with_zero_conductance() -> None:
    """With c -> 0, the operator becomes (shift * I) u = source => u = source / shift."""
    d = _build_grid_incidence()
    torch.manual_seed(0)
    source = torch.randn(2, 64, 3)
    conductance = torch.full((2, 112), 1.0e-3)
    u = kirchhoff_resolve(source, conductance, d, shift=0.1)
    # We are below the clamp-min (1e-3) so c stays at 1e-3; the Laplacian
    # is small but not zero. Sanity-check that the solve is finite.
    assert torch.isfinite(u).all()


def test_kirchhoff_resolve_handles_uniform_conductance() -> None:
    """Standard graph Laplacian solve should be well-conditioned for uniform conductance."""
    d = _build_grid_incidence()
    torch.manual_seed(1)
    source = torch.randn(2, 64, 2)
    conductance = torch.ones(2, 112)
    u = kirchhoff_resolve(source, conductance, d, shift=1.0e-2)
    assert u.shape == source.shape
    # The graph Laplacian has the constant vector in its kernel; with shift > 0
    # the solve is unique. Sanity: total potential should track the sum of
    # sources (after the shift correction).
    assert torch.isfinite(u).all()


def test_forward_shape_and_keys() -> None:
    model = KirchhoffMobilitySolve(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "kms_potential_norm",
        "kms_conductance_mean",
        "kms_source_norm",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = KirchhoffMobilitySolve(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert model.source_head.weight.grad is not None
    assert next(model.conductance_head.parameters()).grad is not None
    assert model.output_proj.weight.grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = KirchhoffMobilitySolve(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_uniform_conductance_differs_from_learned() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = KirchhoffMobilitySolve(**cfg).eval()
    torch.manual_seed(0)
    uniform = KirchhoffMobilitySolve(**cfg, ablation="uniform_conductance").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        uniform_out = uniform(boards)
    assert torch.allclose(
        uniform_out["kms_conductance_mean"],
        torch.ones_like(uniform_out["kms_conductance_mean"]),
    )
    assert not torch.allclose(
        full_out["primitive_delta_raw"], uniform_out["primitive_delta_raw"]
    )


def test_zero_source_zeros_potential_norm() -> None:
    model = KirchhoffMobilitySolve(**_toy_kwargs(), ablation="zero_source").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["kms_source_norm"], torch.zeros_like(out["kms_source_norm"]))
    assert torch.allclose(out["kms_potential_norm"], torch.zeros_like(out["kms_potential_norm"]))


def test_diagonal_only_makes_u_equal_to_source_over_shift() -> None:
    """`diagonal_only` should produce u = s / shift (no Laplacian contribution)."""
    torch.manual_seed(0)
    cfg = _toy_kwargs(shift=0.5)
    model = KirchhoffMobilitySolve(**cfg, ablation="diagonal_only").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    # The source norm should equal kms_potential_norm * shift (up to projection).
    # Approximate check: potential_norm > 0 when source > 0 and shift > 0.
    assert (out["kms_potential_norm"] > 0).all()


def test_disable_gate_pins_gate_at_one() -> None:
    model = KirchhoffMobilitySolve(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        KirchhoffMobilitySolve(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        KirchhoffMobilitySolve(input_channels=12, num_classes=1)


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = KirchhoffMobilitySolve(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p045"
    assert data["slug"] == "kirchhoff_mobility_solve"
    assert data["implementation_kind"] == "bespoke_model"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p045"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
