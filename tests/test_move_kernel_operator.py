"""Focused tests for the p033 Move-Kernel Operator primitive (MKO)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.move_kernel_operator import (
    ALLOWED_ABLATIONS,
    MKO_MOVE_TYPES,
    MoveKernelOperator,
    build_move_kernel_operator_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "move_kernel_operator"
IDEA_DIR = Path("ideas/registry/p033_move_kernel_operator")
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
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, MoveKernelOperator)
    aliased = build_move_kernel_operator_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "feature_dim": 8,
            "head_hidden_dim": 16,
        }
    )
    assert aliased.trunk.channels == 12


def test_forward_shape_and_keys() -> None:
    model = MoveKernelOperator(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
    ):
        assert key in out and out[key].shape == (2,), key
    for name, _ in MKO_MOVE_TYPES:
        diag_key = f"mko_norm_{name}"
        assert diag_key in out and out[diag_key].shape == (2,), diag_key


def test_static_masks_are_registered_buffers() -> None:
    model = MoveKernelOperator(**_toy_kwargs())
    assert "move_type_masks" in dict(model.named_buffers())
    masks = model.move_type_masks
    assert masks.shape == (len(MKO_MOVE_TYPES), 64, 64)
    # Knight mask row for b1 (plane row 7, file 1 -> square 57) should have
    # exactly three legal knight targets: a3 (40), c3 (42), and d2 (51).
    knight_row = masks[0, 57]
    assert int(knight_row.sum().item()) == 3
    expected_targets = {40, 42, 51}
    nonzero = set(torch.nonzero(knight_row).view(-1).tolist())
    assert nonzero == expected_targets


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = MoveKernelOperator(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    for proj in model.type_projections:
        assert proj.weight.grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    # `type_scalars` is exercised only under the `scalar_per_type` ablation.
    # `shared_projection` is exercised only under the `shared_kernel` ablation.


def test_scalar_per_type_ablation_has_scalar_grad() -> None:
    model = MoveKernelOperator(**_toy_kwargs(), ablation="scalar_per_type")
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert model.type_scalars.grad is not None
    assert model.type_scalars.grad.abs().sum().item() > 0


def test_zero_delta_recovers_trunk_logit() -> None:
    model = MoveKernelOperator(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_shared_kernel_differs_from_full() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = MoveKernelOperator(**cfg).eval()
    torch.manual_seed(0)
    shared = MoveKernelOperator(**cfg, ablation="shared_kernel").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        shared_out = shared(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], shared_out["primitive_delta_raw"]
    )


def test_scalar_per_type_differs_from_full() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = MoveKernelOperator(**cfg).eval()
    torch.manual_seed(0)
    scalar = MoveKernelOperator(**cfg, ablation="scalar_per_type").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        scalar_out = scalar(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], scalar_out["primitive_delta_raw"]
    )


def test_disable_gate_pins_gate_at_one() -> None:
    model = MoveKernelOperator(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_shuffle_features_remains_finite() -> None:
    torch.manual_seed(0)
    model = MoveKernelOperator(**_toy_kwargs(), ablation="shuffle_features").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.isfinite(out["logits"]).all()


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        MoveKernelOperator(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        MoveKernelOperator(input_channels=12, num_classes=1)


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = MoveKernelOperator(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p033"
    assert data["slug"] == "move_kernel_operator"
    assert data["implementation_kind"] == "bespoke_model"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p033"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
