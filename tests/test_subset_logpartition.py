"""Focused tests for the p046 Bounded Subset Log-Partition primitive (SLPT)."""
from __future__ import annotations

from pathlib import Path

import chess
import math

import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.subset_logpartition import (
    ALLOWED_ABLATIONS,
    SubsetLogPartition,
    build_subset_logpartition_from_config,
    subset_logpartition_scan,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "subset_logpartition"
IDEA_DIR = Path("ideas/registry/p046_subset_logpartition")
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
        "log_weight_dim": 8,
        "degree": 3,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, SubsetLogPartition)
    aliased = build_subset_logpartition_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "log_weight_dim": 6,
            "degree": 2,
            "head_hidden_dim": 16,
        }
    )
    assert aliased.trunk.channels == 12
    assert aliased.degree == 2


def test_subset_logpartition_scan_matches_explicit_enumeration() -> None:
    """Verify the log-semiring scan matches the explicit sum over subsets."""
    torch.manual_seed(0)
    a = torch.randn(2, 5, 3)
    mask = torch.tensor([[1, 1, 0, 1, 1], [0, 1, 1, 1, 0]], dtype=torch.float32)
    out = subset_logpartition_scan(a, mask, degree=2)
    # Reference Y[1] = logsumexp over active singletons.
    weighted = a + torch.where(
        mask.unsqueeze(-1) > 0.5,
        torch.zeros_like(mask).unsqueeze(-1),
        torch.full_like(mask, -1.0e9).unsqueeze(-1),
    )
    ref1 = torch.logsumexp(weighted, dim=1)
    assert torch.allclose(out[:, 0], ref1, atol=1e-4)
    # Reference Y[2] = log sum_{i<j active} exp(a_i + a_j) per channel.
    ref2 = torch.full_like(out[:, 1], -1.0e9)
    n = a.shape[1]
    for b in range(a.shape[0]):
        for c in range(a.shape[-1]):
            logs = []
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[b, i] > 0.5 and mask[b, j] > 0.5:
                        logs.append((weighted[b, i, c] + weighted[b, j, c]).item())
            if logs:
                ref2[b, c] = torch.logsumexp(torch.tensor(logs), dim=0)
    finite_mask = torch.isfinite(out[:, 1]) & torch.isfinite(ref2)
    assert torch.allclose(out[:, 1][finite_mask], ref2[finite_mask], atol=1e-3)


def test_subset_logpartition_scan_order_invariance() -> None:
    """Coefficient values are invariant under token permutation when mask follows."""
    torch.manual_seed(1)
    a = torch.randn(2, 6, 4)
    mask = torch.ones(2, 6)
    base = subset_logpartition_scan(a, mask, degree=3)
    perm = torch.randperm(6)
    permuted = subset_logpartition_scan(a[:, perm], mask[:, perm], degree=3)
    assert torch.allclose(base, permuted, atol=1e-4)


def test_forward_shape_and_keys() -> None:
    model = SubsetLogPartition(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "slpt_active_mean",
        "slpt_logpartition_norm",
        "slpt_y1",
        "slpt_y2",
        "slpt_y3",
    ):
        assert key in out and out[key].shape == (2,), key


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = SubsetLogPartition(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.log_weight_proj.parameters()).grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None


def test_zero_delta_recovers_trunk_logit() -> None:
    model = SubsetLogPartition(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_k1_only_differs_from_full_K() -> None:
    cfg = _toy_kwargs(degree=3)
    torch.manual_seed(0)
    full = SubsetLogPartition(**cfg).eval()
    torch.manual_seed(0)
    k1 = SubsetLogPartition(**cfg, ablation="k1_only").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        k1_out = k1(boards)
    assert not torch.allclose(full_out["primitive_delta_raw"], k1_out["primitive_delta_raw"])


def test_uniform_mask_differs_from_occupancy_mask() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = SubsetLogPartition(**cfg).eval()
    torch.manual_seed(0)
    uniform = SubsetLogPartition(**cfg, ablation="uniform_mask").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        uniform_out = uniform(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], uniform_out["primitive_delta_raw"]
    )


def test_disable_gate_pins_gate_at_one() -> None:
    model = SubsetLogPartition(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_shuffle_tokens_preserves_delta() -> None:
    """Token-shuffle ablation should produce identical output via invariance."""
    torch.manual_seed(0)
    cfg = _toy_kwargs(degree=2)
    full = SubsetLogPartition(**cfg).eval()
    torch.manual_seed(0)
    shuffled = SubsetLogPartition(**cfg, ablation="shuffle_tokens").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        shuffled_out = shuffled(boards)
    assert torch.allclose(
        full_out["primitive_delta_raw"], shuffled_out["primitive_delta_raw"], atol=1e-4
    )


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        SubsetLogPartition(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        SubsetLogPartition(input_channels=12, num_classes=1)


def test_rejects_invalid_degree() -> None:
    with pytest.raises(ValueError):
        SubsetLogPartition(**_toy_kwargs(degree=0))
    with pytest.raises(ValueError):
        SubsetLogPartition(**_toy_kwargs(degree=6))


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = SubsetLogPartition(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p046"
    assert data["slug"] == "subset_logpartition"
    assert data["implementation_kind"] == "bespoke_model"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p046"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
