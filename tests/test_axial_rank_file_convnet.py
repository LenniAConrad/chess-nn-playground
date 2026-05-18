"""Focused tests for the bespoke i149 Axial Rank-File ConvNet."""
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
from chess_nn_playground.models.trunk.axial_rank_file_convnet import (
    AxialRankFileBlock,
    AxialRankFileConvNet,
    AxialRankFileHead,
    AxialRankFileStem,
    build_axial_rank_file_convnet_from_config,
)


REGISTRY_KEY = "axial_rank_file_convnet"
IDEA_DIR = Path("ideas/registry/i149_axial_rank_file_convnet")
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_PUZZLE_FEN = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 8,
        "depth": 2,
        "hidden_dim": 16,
        "dropout": 0.0,
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
    assert isinstance(model, AxialRankFileConvNet)
    assert not isinstance(model, ResearchPacketProbe)


def test_builder_passes_through_blocks_alias() -> None:
    model = build_axial_rank_file_convnet_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 8,
            "blocks": 1,
            "hidden_dim": 16,
            "dropout": 0.0,
            "use_batchnorm": False,
        }
    )
    assert model.depth == 1


def test_forward_shape_and_required_keys() -> None:
    model = AxialRankFileConvNet(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "trunk_energy",
        "rank_energy",
        "file_energy",
        "local_energy",
        "axial_balance",
        "rank_pool_norm",
        "file_pool_norm",
        "global_pool_norm",
        "rank_file_imbalance",
        "piece_density",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "axial_rank_file_ablation",
        "axial_rank_file_block_count",
    }
    assert expected_keys.issubset(out)
    for key in expected_keys - {"logits", "prob"}:
        assert out[key].shape[0] == 2, key


def test_prob_is_in_unit_interval() -> None:
    model = AxialRankFileConvNet(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["prob"].shape == (2,)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()


def test_backward_gradients_flow_through_stem_and_head() -> None:
    model = AxialRankFileConvNet(**_toy_kwargs())
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    has_stem_grad = any(
        (p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in model.stem.parameters()
    )
    assert has_stem_grad
    head_first = model.head.classifier[1].weight.grad
    assert head_first is not None and torch.isfinite(head_first).all()


def test_local_only_zeros_rank_and_file_energy() -> None:
    model = AxialRankFileConvNet(**_toy_kwargs(), ablation="local_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["rank_energy"], torch.zeros_like(out["rank_energy"]))
    assert torch.allclose(out["file_energy"], torch.zeros_like(out["file_energy"]))
    assert out["local_energy"].abs().sum() > 0


def test_rank_only_zeros_file_and_local_energy() -> None:
    model = AxialRankFileConvNet(**_toy_kwargs(), ablation="rank_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["file_energy"], torch.zeros_like(out["file_energy"]))
    assert torch.allclose(out["local_energy"], torch.zeros_like(out["local_energy"]))
    assert out["rank_energy"].abs().sum() > 0


def test_file_only_zeros_rank_and_local_energy() -> None:
    model = AxialRankFileConvNet(**_toy_kwargs(), ablation="file_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["rank_energy"], torch.zeros_like(out["rank_energy"]))
    assert torch.allclose(out["local_energy"], torch.zeros_like(out["local_energy"]))
    assert out["file_energy"].abs().sum() > 0


def test_single_block_collapses_depth() -> None:
    kwargs = _toy_kwargs(depth=4)
    model = AxialRankFileConvNet(**kwargs, ablation="single_block").eval()
    assert model.effective_depth == 1
    assert model.depth == 4


def test_all_ablations_run_without_crash() -> None:
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    for ablation in AxialRankFileConvNet.ABLATIONS:
        torch.manual_seed(0)
        model = AxialRankFileConvNet(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation
        assert out["axial_rank_file_ablation"][0].item() == float(
            AxialRankFileConvNet.ABLATIONS.index(ablation)
        )


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        AxialRankFileConvNet(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    model = AxialRankFileConvNet(**_toy_kwargs())
    bad = torch.zeros(2, 12, 8, 8)
    with pytest.raises(ValueError):
        model(bad)


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "i149"
    assert data["slug"] == "axial_rank_file_convnet"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "i149"
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
    assert isinstance(model, AxialRankFileConvNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.stem, AxialRankFileStem)
    assert isinstance(model.head, AxialRankFileHead)
    assert all(isinstance(block, AxialRankFileBlock) for block in model.blocks)

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

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i149"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues, conformance_rows[0].issues
