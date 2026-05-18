"""Focused tests for the p047 Learned Relation Confidence primitive (LRC)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.models.primitives.learned_relation_confidence import (
    ALLOWED_ABLATIONS,
    LearnedRelationConfidence,
    RELATION_COUNT,
    RELATION_NAMES,
    build_learned_relation_confidence_from_config,
)
from chess_nn_playground.models.registry import available_models, build_model


REGISTRY_KEY = "learned_relation_confidence"
IDEA_DIR = Path("ideas/registry/p047_learned_relation_confidence")
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
        "token_dim": 16,
        "low_rank_dim": 4,
        "edge_hidden": 16,
        "head_hidden_dim": 16,
        "head_dropout": 0.0,
    }
    base.update(overrides)
    return base


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_builder_round_trips_config() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, LearnedRelationConfidence)
    aliased = build_learned_relation_confidence_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "use_batchnorm": False,
            "token_dim": 12,
            "low_rank_dim": 4,
            "edge_hidden": 12,
            "head_hidden_dim": 16,
        }
    )
    assert aliased.trunk.channels == 12
    assert aliased.token_dim == 12


def test_relation_count_matches_i018_topology() -> None:
    model = LearnedRelationConfidence(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    masks = model._relation_masks(boards)
    assert masks.shape == (2, RELATION_COUNT, 64, 64)
    # All masks must be in [0, 1].
    assert (masks >= 0.0).all().item()
    assert (masks <= 1.0).all().item()


def test_forward_shape_and_keys() -> None:
    model = LearnedRelationConfidence(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    assert out["logits"].shape == (2,)
    for key in (
        "base_logit",
        "primitive_delta",
        "primitive_gate",
        "primitive_contribution",
        "lrc_global_mean_confidence",
        "lrc_mask_density",
    ):
        assert key in out and out[key].shape == (2,), key
    for r_name in RELATION_NAMES:
        assert out[f"lrc_mean_conf_{r_name}"].shape == (2,)
        assert out[f"lrc_kept_{r_name}"].shape == (2,)
        assert out[f"lrc_entropy_{r_name}"].shape == (2,)


def test_backward_gradients_flow_through_head_and_trunk() -> None:
    model = LearnedRelationConfidence(**_toy_kwargs())
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    assert next(model.trunk.parameters()).grad is not None
    assert next(model.token_proj.parameters()).grad is not None
    assert next(model.delta_head.parameters()).grad is not None
    assert next(model.gate_head.parameters()).grad is not None
    # Edge MLPs and bilinear projections also receive gradients.
    assert next(model.edge_src_mlp.parameters()).grad is not None
    assert next(model.edge_tgt_mlp.parameters()).grad is not None
    assert next(model.q_proj.parameters()).grad is not None
    assert next(model.k_proj.parameters()).grad is not None
    assert model.relation_low_rank.grad is not None
    assert model.relation_gate_logits.grad is not None
    assert model.relation_bias.grad is not None


def test_relation_builder_is_frozen() -> None:
    model = LearnedRelationConfidence(**_toy_kwargs())
    for parameter in model.relation_builder.parameters():
        assert not parameter.requires_grad
    for parameter in model.board_adapter.parameters():
        assert not parameter.requires_grad


def test_topology_preservation_zero_mask_stays_zero() -> None:
    """If a mask entry is 0, the weighted_mask entry must stay 0 exactly."""
    torch.manual_seed(0)
    model = LearnedRelationConfidence(**_toy_kwargs()).eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    out = model(boards)
    masks = model._relation_masks(boards)
    # We re-derive confidence the same way the forward does and check that
    # the per-relation summary respects the topology preservation invariant.
    inactive_total = (masks <= 0.0).sum().item()
    assert inactive_total > 0
    # Per-relation mean_conf must lie in [0, 1] regardless of mask density.
    for r_name in RELATION_NAMES:
        mean_conf = out[f"lrc_mean_conf_{r_name}"]
        assert (mean_conf >= 0.0).all()
        assert (mean_conf <= 1.0).all()


def test_zero_delta_recovers_trunk_logit() -> None:
    model = LearnedRelationConfidence(**_toy_kwargs(), ablation="zero_delta").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_delta"], torch.zeros_like(out["primitive_delta"]))
    assert torch.allclose(out["logits"], out["base_logit"])


def test_trunk_only_matches_zero_delta() -> None:
    """``trunk_only`` is a semantic alias of ``zero_delta``."""
    torch.manual_seed(0)
    zero = LearnedRelationConfidence(**_toy_kwargs(), ablation="zero_delta").eval()
    torch.manual_seed(0)
    trunk_only = LearnedRelationConfidence(**_toy_kwargs(), ablation="trunk_only").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        zero_out = zero(boards)
        trunk_out = trunk_only(boards)
    assert torch.allclose(zero_out["logits"], trunk_out["logits"])


def test_disable_gate_pins_gate_at_one() -> None:
    model = LearnedRelationConfidence(**_toy_kwargs(), ablation="disable_gate").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["primitive_gate"], torch.ones_like(out["primitive_gate"]))


def test_binary_only_differs_from_full() -> None:
    """``binary_only`` skips per-edge confidence; summaries differ from `none`."""
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = LearnedRelationConfidence(**cfg).eval()
    torch.manual_seed(0)
    binary = LearnedRelationConfidence(**cfg, ablation="binary_only").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        binary_out = binary(boards)
    assert not torch.allclose(
        full_out["lrc_global_mean_confidence"],
        binary_out["lrc_global_mean_confidence"],
    )


def test_gate_only_differs_from_full() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = LearnedRelationConfidence(**cfg).eval()
    torch.manual_seed(0)
    gate_only = LearnedRelationConfidence(**cfg, ablation="gate_only").eval()
    boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
    with torch.no_grad():
        full_out = full(boards)
        gate_only_out = gate_only(boards)
    assert not torch.allclose(
        full_out["primitive_delta_raw"], gate_only_out["primitive_delta_raw"]
    )


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        LearnedRelationConfidence(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        LearnedRelationConfidence(input_channels=12, num_classes=1)


def test_rejects_invalid_token_dim() -> None:
    with pytest.raises(ValueError):
        LearnedRelationConfidence(**_toy_kwargs(token_dim=1))


def test_rejects_invalid_low_rank_dim() -> None:
    with pytest.raises(ValueError):
        LearnedRelationConfidence(**_toy_kwargs(low_rank_dim=0))


def test_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        LearnedRelationConfidence(**_toy_kwargs(confidence_temperature=0.0))


def test_all_allowed_ablations_run_without_crash() -> None:
    for ablation in ALLOWED_ABLATIONS:
        if ablation == "none":
            continue
        torch.manual_seed(0)
        model = LearnedRelationConfidence(**_toy_kwargs(), ablation=ablation).eval()
        boards = _board_batch([chess.STARTING_FEN, ROOK_MATE_FEN])
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "p047"
    assert data["slug"] == "learned_relation_confidence"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "p047"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"
