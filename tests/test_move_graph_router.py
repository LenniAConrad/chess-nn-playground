"""Focused tests for the p006 Move-Graph Router primitive."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.move_graph_router import (
    ALLOWED_ABLATIONS,
    MoveGraphRouter,
    build_move_graph_router_from_config,
)
from chess_nn_playground.models.primitives.rule_graph_features import (
    compute_legal_move_graph,
    rule_geometry,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "move_graph_router"
IDEA_DIR = Path("ideas/registry/p006_move_graph_router")

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TACTICAL_FEN = "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"
EMPTY_OWN_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


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
        "token_hidden_dim": 0,
        "edge_hidden_dim": 24,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
        "gate_init": -2.0,
    }
    base.update(overrides)
    return base


def _fen_to_tensor(fen: str) -> torch.Tensor:
    return torch.from_numpy(fen_to_simple_18(fen)).unsqueeze(0).float()


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(f) for f in fens])).float()


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, MoveGraphRouter)


def test_forward_shapes_and_keys() -> None:
    model = MoveGraphRouter(**_toy_kwargs()).eval()
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["base_logit"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    assert out["primitive_gate"].shape == (2,)
    assert out["mgr_edge_count"].shape == (2,)
    assert torch.all(out["primitive_gate"] >= 0.0)
    assert torch.all(out["primitive_gate"] <= 1.0)


def test_edge_count_matches_rule_derived_legal_graph() -> None:
    geom = rule_geometry()
    model = MoveGraphRouter(**_toy_kwargs()).eval()
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    legal = compute_legal_move_graph(boards, geom)
    expected = legal.sum(dim=(1, 2))
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["mgr_edge_count"], expected)


def test_zero_delta_recovers_base_logit() -> None:
    model = MoveGraphRouter(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_disables_primitive_completely() -> None:
    model = MoveGraphRouter(**_toy_kwargs(), ablation="trunk_only").eval()
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))


def test_disable_gate_holds_gate_at_one() -> None:
    model = MoveGraphRouter(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_dense_edges_ablation_uses_full_mask() -> None:
    model = MoveGraphRouter(**_toy_kwargs(), ablation="dense_edges").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    # dense edges = 64*64 = 4096
    assert out["mgr_edge_count"][0].item() == 4096.0


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = MoveGraphRouter(**_toy_kwargs())
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    out = model(boards)
    loss = out["logits"].pow(2).mean()
    loss.backward()
    trunk_param = next(model.trunk.parameters())
    edge_param = next(model.edge_mlp.parameters())
    gate_param = next(model.gate_head.parameters())
    assert trunk_param.grad is not None
    assert edge_param.grad is not None
    assert gate_param.grad is not None
    assert trunk_param.grad.abs().sum().item() > 0
    assert edge_param.grad.abs().sum().item() > 0


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        MoveGraphRouter(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input_at_init() -> None:
    with pytest.raises(ValueError):
        MoveGraphRouter(input_channels=12, num_classes=1)


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        MoveGraphRouter(input_channels=18, num_classes=3)


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
        "edge_hidden_dim": 24,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_move_graph_router_from_config(cfg)
    assert isinstance(model, MoveGraphRouter)
    assert model.trunk.channels == 24


def test_idea_config_yaml_matches_registry_key() -> None:
    config_path = IDEA_DIR / "config.yaml"
    if not config_path.exists():
        pytest.skip("Idea config not present in worktree.")
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["idea_id"] == "p006"
    assert ALLOWED_ABLATIONS == (
        "none",
        "random_edges",
        "dense_edges",
        "zero_delta",
        "disable_gate",
        "trunk_only",
    )
