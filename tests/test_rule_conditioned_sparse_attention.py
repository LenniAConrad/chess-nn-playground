"""Focused tests for the p008 Rule-Conditioned Sparse Attention (MobScan) primitive."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.rule_conditioned_sparse_attention import (
    ALLOWED_ABLATIONS,
    RuleConditionedSparseAttention,
    build_rule_conditioned_sparse_attention_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "rule_conditioned_sparse_attention"
IDEA_DIR = Path("ideas/registry/p008_rule_conditioned_sparse_attention")

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
        "state_dim": 16,
        "num_iterations": 2,
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
    assert isinstance(model, RuleConditionedSparseAttention)


def test_forward_shapes_and_keys() -> None:
    model = RuleConditionedSparseAttention(**_toy_kwargs()).eval()
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    assert out["primitive_delta"].shape == (2,)
    assert out["primitive_gate"].shape == (2,)
    assert out["mobscan_edge_count"].shape == (2,)
    assert out["mobscan_state_norm"].shape == (2,)
    assert out["mobscan_gate_A_mean"].shape == (2,)
    assert out["mobscan_gate_B_mean"].shape == (2,)
    assert out["mobscan_gate_C_mean"].shape == (2,)


def test_iterations_property_obeys_ablation() -> None:
    model = RuleConditionedSparseAttention(**_toy_kwargs(num_iterations=3))
    assert model.num_iterations == 3
    model_single = RuleConditionedSparseAttention(
        **_toy_kwargs(num_iterations=3, ablation="single_iteration")
    )
    assert model_single.num_iterations == 1


def test_zero_delta_recovers_base_logit() -> None:
    model = RuleConditionedSparseAttention(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_untied_state_holds_gates_at_constants() -> None:
    model = RuleConditionedSparseAttention(**_toy_kwargs(), ablation="untied_state").eval()
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["mobscan_gate_A_mean"], torch.full_like(out["mobscan_gate_A_mean"], 0.5))
    assert torch.allclose(out["mobscan_gate_B_mean"], torch.full_like(out["mobscan_gate_B_mean"], 0.5))
    assert torch.allclose(out["mobscan_gate_C_mean"], torch.full_like(out["mobscan_gate_C_mean"], 1.0))


def test_dense_edges_ablation_uses_full_mask() -> None:
    model = RuleConditionedSparseAttention(**_toy_kwargs(), ablation="dense_edges").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["mobscan_edge_count"][0].item() == 4096.0


def test_disable_gate_holds_gate_at_one() -> None:
    model = RuleConditionedSparseAttention(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([INITIAL_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_backward_gradients_flow_through_trunk_and_head() -> None:
    model = RuleConditionedSparseAttention(**_toy_kwargs())
    boards = _board_batch([INITIAL_FEN, TACTICAL_FEN])
    out = model(boards)
    loss = out["logits"].pow(2).mean()
    loss.backward()
    trunk_param = next(model.trunk.parameters())
    gate_param = next(model.gate_head.parameters())
    A_param = model.gate_A.weight
    assert trunk_param.grad is not None
    assert gate_param.grad is not None
    assert A_param.grad is not None
    assert trunk_param.grad.abs().sum().item() > 0


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        RuleConditionedSparseAttention(**_toy_kwargs(), ablation="not_a_real_ablation")


def test_rejects_non_simple_18_input_at_init() -> None:
    with pytest.raises(ValueError):
        RuleConditionedSparseAttention(input_channels=12, num_classes=1)


def test_rejects_zero_iterations() -> None:
    with pytest.raises(ValueError):
        RuleConditionedSparseAttention(**_toy_kwargs(num_iterations=0))


def test_rejects_multiclass_head() -> None:
    with pytest.raises(ValueError):
        RuleConditionedSparseAttention(input_channels=18, num_classes=3)


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
        "state_dim": 16,
        "num_iterations": 2,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    model = build_rule_conditioned_sparse_attention_from_config(cfg)
    assert isinstance(model, RuleConditionedSparseAttention)
    assert model.trunk.channels == 24


def test_idea_config_yaml_matches_registry_key() -> None:
    config_path = IDEA_DIR / "config.yaml"
    if not config_path.exists():
        pytest.skip("Idea config not present in worktree.")
    with config_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["idea_id"] == "p008"
    assert set(ALLOWED_ABLATIONS) >= {"none", "random_edges", "untied_state", "single_iteration"}
