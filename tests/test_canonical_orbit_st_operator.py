"""Tests for the p036 Canonical-Orbit Straight-Through Operator primitive."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.canonical_orbit_st_operator import (
    CanonicalOrbitSTOperator,
    _build_permutations,
    build_canonical_orbit_st_operator_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


def _toy_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 32,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "latent_dim": 8,
        "key_dim": 4,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "canonical_orbit_st_operator" in available_models()
    model = build_model("canonical_orbit_st_operator", _toy_kwargs())
    assert isinstance(model, CanonicalOrbitSTOperator)


def test_permutations_are_valid_involutions() -> None:
    perms = _build_permutations()
    assert perms.shape == (4, 64)
    # Each row is a permutation of 0..63.
    for g in range(4):
        sorted_row = torch.sort(perms[g]).values
        assert torch.equal(sorted_row, torch.arange(64))
    # Each element of C2 x C2 is its own inverse: perm[perm] = identity.
    identity = torch.arange(64)
    for g in range(4):
        composed = perms[g][perms[g]]
        assert torch.equal(composed, identity), f"perm {g} not an involution"


def test_file_mirror_permutation_swaps_columns() -> None:
    perms = _build_permutations()
    file_mirror = perms[1].view(8, 8)
    # The first row should map column j to column 7-j (and similarly for all rows).
    for row in range(8):
        for col in range(8):
            assert file_mirror[row, col].item() == row * 8 + (7 - col)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = CanonicalOrbitSTOperator(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    assert out["logits"].shape == (1,)
    for diag in (
        "base_logit",
        "primitive_delta",
        "primitive_delta_raw",
        "primitive_gate",
        "primitive_gate_logit",
        "cost_chosen_orbit_index",
        "cost_orbit_gap",
        "cost_orbit_ties",
        "cost_residual_norm",
        "cost_canonical_norm",
    ):
        assert diag in out, f"missing diagnostic: {diag}"


def test_chosen_orbit_index_is_in_range() -> None:
    model = CanonicalOrbitSTOperator(**_toy_kwargs()).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    chosen = out["cost_chosen_orbit_index"].long()
    assert chosen.shape == (2,)
    assert ((chosen >= 0) & (chosen < 4)).all()


def test_starting_position_chooses_a_consistent_orbit() -> None:
    # The starting position is symmetric under file mirror, so the operator
    # should produce a tie >= 2 for that batch row.
    model = CanonicalOrbitSTOperator(**_toy_kwargs()).eval()
    # Disable randomness in BatchNorm (already off via use_batchnorm=False).
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    # The starting position projects into a latent that may or may not
    # produce an exact tie because the latent projection is *not* file-mirror
    # equivariant. We assert that the tie count is at least 1 (the trivial
    # tie with itself) and that the chosen orbit is deterministic across
    # repeat calls.
    assert out["cost_orbit_ties"][0].item() >= 1
    with torch.no_grad():
        out_repeat = model(boards)
    assert torch.equal(out["cost_chosen_orbit_index"], out_repeat["cost_chosen_orbit_index"])


def test_identity_only_ablation_always_picks_identity() -> None:
    model = CanonicalOrbitSTOperator(**_toy_kwargs(ablation="identity_only")).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    chosen = out["cost_chosen_orbit_index"].long()
    assert torch.equal(chosen, torch.zeros(2, dtype=torch.long))


def test_fixed_choice_ablation_always_picks_file_mirror() -> None:
    model = CanonicalOrbitSTOperator(**_toy_kwargs(ablation="fixed_choice")).eval()
    boards = _board_batch([
        chess.STARTING_FEN,
        "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
    ])
    with torch.no_grad():
        out = model(boards)
    chosen = out["cost_chosen_orbit_index"].long()
    assert torch.equal(chosen, torch.ones(2, dtype=torch.long))


def test_zero_delta_ablation_recovers_trunk_logit() -> None:
    model = CanonicalOrbitSTOperator(**_toy_kwargs(ablation="zero_delta")).eval()
    boards = _board_batch([chess.STARTING_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = CanonicalOrbitSTOperator(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad.abs().sum() > 0
    assert model.latent_proj.weight.grad.abs().sum() > 0


def test_hash_projection_is_deterministic_across_instances() -> None:
    model_a = CanonicalOrbitSTOperator(**_toy_kwargs())
    model_b = CanonicalOrbitSTOperator(**_toy_kwargs())
    # Hash projection is a buffer derived from a fixed seed.
    assert torch.equal(model_a.hash_projection, model_b.hash_projection)
    assert torch.equal(model_a.hash_square_weights, model_b.hash_square_weights)


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "latent_dim": 8,
        "key_dim": 4,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_canonical_orbit_st_operator_from_config(cfg)
    assert isinstance(model, CanonicalOrbitSTOperator)
    assert model.trunk.channels == 24


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        CanonicalOrbitSTOperator(**_toy_kwargs(ablation="totally_not_real"))


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        CanonicalOrbitSTOperator(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        CanonicalOrbitSTOperator(input_channels=18, num_classes=3)
