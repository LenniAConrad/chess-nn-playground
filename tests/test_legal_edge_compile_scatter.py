"""Focused tests for the p011 Legal-Edge Compile Scatter primitive."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.legal_edge_compile_scatter import (
    ALLOWED_ABLATIONS,
    LegalEdgeCompileScatter,
    build_legal_edge_compile_scatter_from_config,
)
from chess_nn_playground.models.primitives.legal_move_graph_delta import (
    PIECE_TYPE_NAMES,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "legal_edge_compile_scatter"
IDEA_DIR = Path("ideas/registry/p011_legal_edge_compile_scatter")

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TACTICAL_FEN = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base = {
        "input_channels": 18,
        "num_classes": 1,
        "trunk_channels": 16,
        "trunk_hidden_dim": 24,
        "trunk_depth": 1,
        "trunk_dropout": 0.0,
        "trunk_use_batchnorm": False,
        "token_embed_dim": 16,
        "message_dim": 16,
        "edge_gate_hidden": 8,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "gate_init": -2.0,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(f) for f in fens])).float()


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, LegalEdgeCompileScatter)


def test_forward_shapes_and_keys() -> None:
    model = LegalEdgeCompileScatter(**_toy_kwargs()).eval()
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    assert out["primitive_gate"].shape == (2,)
    assert out["lecs_edge_count"].shape == (2,)
    assert out["lecs_gate_mean"].shape == (2,)
    for name in PIECE_TYPE_NAMES:
        key = f"lecs_msg_norm_{name}"
        assert key in out
        assert out[key].shape == (2,)


def test_gate_mean_bounded_in_unit_interval() -> None:
    model = LegalEdgeCompileScatter(**_toy_kwargs()).eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["lecs_gate_mean"][0].item() >= 0.0
    assert out["lecs_gate_mean"][0].item() <= 1.0 + 1e-5


def test_no_edge_gate_holds_gate_at_one_on_valid_edges() -> None:
    # In the no_edge_gate ablation the gate equals the mask; mean over valid
    # edges is 1.0.
    model = LegalEdgeCompileScatter(**_toy_kwargs(), ablation="no_edge_gate").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["lecs_gate_mean"][0].item() == pytest.approx(1.0, abs=1e-5)


def test_zero_delta_recovers_base_logit() -> None:
    model = LegalEdgeCompileScatter(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_shared_type_weight_collapses_per_type_projection() -> None:
    model_shared = LegalEdgeCompileScatter(**_toy_kwargs(), ablation="shared_type_weight").eval()
    assert model_shared.message_linear is not None
    assert model_shared.message_linears is None


def test_disable_gate_holds_gate_at_one() -> None:
    model = LegalEdgeCompileScatter(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = LegalEdgeCompileScatter(**_toy_kwargs())
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    out = model(boards)
    loss = out["logits"].pow(2).mean()
    loss.backward()
    trunk_param = next(model.trunk.parameters())
    edge_gate_param = next(model.edge_gate_mlps[0].parameters())
    gate_param = next(model.gate_head.parameters())
    assert trunk_param.grad is not None
    assert edge_gate_param.grad is not None
    assert gate_param.grad is not None


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        LegalEdgeCompileScatter(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input_at_init() -> None:
    with pytest.raises(ValueError):
        LegalEdgeCompileScatter(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        LegalEdgeCompileScatter(input_channels=18, num_classes=3)


def test_builder_accepts_aliased_channel_keys() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 24,
        "hidden_dim": 48,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "token_embed_dim": 16,
        "message_dim": 16,
        "edge_gate_hidden": 8,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_legal_edge_compile_scatter_from_config(cfg)
    assert isinstance(model, LegalEdgeCompileScatter)
    assert model.trunk.channels == 24


def test_idea_config_yaml_matches_registry_key() -> None:
    config_path = IDEA_DIR / "config.yaml"
    if not config_path.exists():
        pytest.skip("Idea config not present in worktree.")
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["idea_id"] == "p011"
    assert set(ALLOWED_ABLATIONS) >= {"none", "no_edge_gate", "random_typed_edges"}
