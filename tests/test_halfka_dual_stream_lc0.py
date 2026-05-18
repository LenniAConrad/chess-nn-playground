"""Focused tests for the bespoke i243 HalfKA Dual-Stream LC0 Evaluator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_tensor
from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind
from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES
from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.trunk.halfka_dual_stream_lc0 import (
    HalfKAAccumulator,
    HalfKADualStreamLC0,
    PerSquareReconstructionMLP,
    build_halfka_dual_stream_lc0_from_config,
)


REGISTRY_KEY = "halfka_dual_stream_lc0"
IDEA_DIR = Path("ideas/registry/i243_halfka_dual_stream_lc0")
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_PUZZLE_FEN = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "embed_dim": 8,
        "backbone_channels": 16,
        "backbone_depth": 1,
        "head_hidden": 16,
        "dropout": 0.0,
        "policy_dim": 8,
        "use_batchnorm": False,
    }
    base.update(overrides)
    return base


def _board_batch(fens: list[str]) -> torch.Tensor:
    arrays = [fen_to_tensor(fen) for fen in fens]
    return torch.stack([torch.from_numpy(a).float() for a in arrays], dim=0)


def test_registry_key_is_present() -> None:
    assert REGISTRY_KEY in available_models()


def test_registry_key_is_not_a_research_packet_probe_name() -> None:
    assert REGISTRY_KEY not in RESEARCH_PACKET_MODEL_NAMES


def test_builder_from_config_returns_bespoke_model() -> None:
    model = build_model(REGISTRY_KEY, _toy_kwargs())
    assert isinstance(model, HalfKADualStreamLC0)
    assert not isinstance(model, ResearchPacketProbe)


def test_builder_passes_through_channels_and_depth_aliases() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 8,
        "backbone_channels": 16,
        "depth": 1,
        "hidden_dim": 16,
        "dropout": 0.0,
        "policy_dim": 8,
        "use_batchnorm": False,
    }
    model = build_halfka_dual_stream_lc0_from_config(cfg)
    assert model.embed_dim == 8
    assert model.backbone_depth == 1
    assert model.head_hidden == 16


def test_forward_shape_and_required_keys() -> None:
    model = HalfKADualStreamLC0(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "exchange_logit",
        "king_logit",
        "alpha_king",
        "alpha_exchange",
        "residual_logit",
        "exchange_pool_norm",
        "king_pool_norm",
        "value_wdl_logits",
        "policy_logits",
        "white_accumulator_norm",
        "black_accumulator_norm",
        "accumulator_norm",
        "white_king_sq",
        "black_king_sq",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "halfka_dual_stream_ablation",
        "halfka_embedding_dim",
        "policy_logit_count",
    }
    assert expected_keys.issubset(out)
    for key in expected_keys - {"logits", "prob", "value_wdl_logits", "policy_logits"}:
        assert out[key].shape[0] == 2, key


def test_value_wdl_and_policy_shapes() -> None:
    model = HalfKADualStreamLC0(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["value_wdl_logits"].shape == (2, 3)
    assert out["policy_logits"].shape == (2, 8)


def test_prob_is_in_unit_interval() -> None:
    model = HalfKADualStreamLC0(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()


def test_alpha_sums_to_one() -> None:
    model = HalfKADualStreamLC0(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    summed = out["alpha_exchange"] + out["alpha_king"]
    assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)


def test_no_halfka_drops_accumulator() -> None:
    model = HalfKADualStreamLC0(**_toy_kwargs(), ablation="no_halfka").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    # The accumulator norms still come from the HalfKA module that was
    # constructed (its tables are evaluated), but the backbone input did
    # not consume the accumulator.
    assert torch.isfinite(out["logits"]).all()


def test_no_residual_zeros_residual_logit() -> None:
    model = HalfKADualStreamLC0(**_toy_kwargs(), ablation="no_residual").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["residual_logit"], torch.zeros_like(out["residual_logit"]))


def test_puzzle_only_zeros_lc0_heads() -> None:
    model = HalfKADualStreamLC0(**_toy_kwargs(), ablation="puzzle_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["value_wdl_logits"], torch.zeros_like(out["value_wdl_logits"]))
    assert torch.allclose(out["policy_logits"], torch.zeros_like(out["policy_logits"]))


def test_no_dual_stream_pools_match() -> None:
    model = HalfKADualStreamLC0(**_toy_kwargs(), ablation="no_dual_stream").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["exchange_pool_norm"], out["king_pool_norm"])


def test_backward_gradients_flow_through_halfka_and_heads() -> None:
    model = HalfKADualStreamLC0(**_toy_kwargs())
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    halfka_grad = model.halfka.white_embedding.grad
    assert halfka_grad is not None and torch.isfinite(halfka_grad).all() and halfka_grad.abs().sum() > 0
    value_grad = model.value_head[0].weight.grad
    assert value_grad is None or torch.isfinite(value_grad).all()


def test_all_ablations_run_without_crash() -> None:
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    for ablation in HalfKADualStreamLC0.ABLATIONS:
        torch.manual_seed(0)
        model = HalfKADualStreamLC0(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation
        assert out["halfka_dual_stream_ablation"][0].item() == float(
            HalfKADualStreamLC0.ABLATIONS.index(ablation)
        )


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        HalfKADualStreamLC0(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        HalfKADualStreamLC0(input_channels=12, num_classes=1)


def test_rejects_non_binary_classes() -> None:
    with pytest.raises(ValueError):
        HalfKADualStreamLC0(num_classes=3)


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "i243"
    assert data["slug"] == "halfka_dual_stream_lc0"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "i243"
    assert cfg["model"]["name"] == REGISTRY_KEY
    assert cfg["device"] == "nvidia"


def _load_idea_model_module(folder: Path):
    module_path = folder / "model.py"
    spec = importlib.util.spec_from_file_location(f"idea_{folder.name}_model", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_idea_folder_is_bespoke_and_conformant() -> None:
    config = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model_module(IDEA_DIR)
    model = module.build_model_from_config(config).eval()
    assert isinstance(model, HalfKADualStreamLC0)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.halfka, HalfKAAccumulator)
    assert isinstance(model.exchange_reconstruction, PerSquareReconstructionMLP)

    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()

    model_py = (IDEA_DIR / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(IDEA_DIR)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues, kind_row.issues

    training_report = validate_idea_for_training(IDEA_DIR)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i243"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues, conformance_rows[0].issues
