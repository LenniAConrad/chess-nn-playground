"""Focused tests for the bespoke i144 Board FPN CNN."""
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
from chess_nn_playground.models.trunk.board_fpn_cnn import (
    BoardFPNCNN,
    BoardFPNCoordinatePlanes,
    BoardFPNConvStack,
    BoardFPNHead,
    build_board_fpn_cnn_from_config,
)


REGISTRY_KEY = "board_fpn_cnn"
IDEA_DIR = Path("ideas/registry/i144_board_fpn_cnn")
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_PUZZLE_FEN = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "width": 8,
        "blocks_per_level": 1,
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
    assert isinstance(model, BoardFPNCNN)
    assert not isinstance(model, ResearchPacketProbe)


def test_builder_passes_through_channels_and_depth_aliases() -> None:
    model = build_board_fpn_cnn_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "depth": 1,
            "hidden_dim": 24,
            "dropout": 0.0,
            "use_batchnorm": False,
        }
    )
    assert model.width == 12


def test_forward_shape_and_required_keys() -> None:
    model = BoardFPNCNN(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "fpn_y8_energy",
        "fpn_y4_energy",
        "fpn_x2_energy",
        "topdown_4_energy",
        "topdown_8_energy",
        "piece_density",
        "coordinate_energy",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "board_fpn_ablation",
        "board_fpn_level_count",
    }
    assert expected_keys.issubset(out)
    for key in expected_keys - {"logits", "prob"}:
        assert out[key].shape[0] == 2, key


def test_prob_is_in_unit_interval() -> None:
    model = BoardFPNCNN(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["prob"].shape == (2,)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()


def test_backward_gradients_flow_through_level_stacks_and_head() -> None:
    model = BoardFPNCNN(**_toy_kwargs())
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    # All conv stacks should have gradients.
    has_conv_grad = any(
        (p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in model.level8.parameters()
    )
    assert has_conv_grad
    head_first = model.head.classifier[1].weight.grad
    assert head_first is not None and torch.isfinite(head_first).all()


def test_coordinate_planes_are_buffers_not_parameters() -> None:
    model = BoardFPNCNN(**_toy_kwargs())
    parameter_ids = {id(p) for p in model.parameters()}
    assert model.coordinates is not None
    assert id(model.coordinates.planes) not in parameter_ids


def test_single_resolution_matched_zeros_fine_levels() -> None:
    model = BoardFPNCNN(**_toy_kwargs(), ablation="single_resolution_matched").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["fpn_y4_energy"], torch.zeros_like(out["fpn_y4_energy"]))
    assert torch.allclose(out["fpn_x2_energy"], torch.zeros_like(out["fpn_x2_energy"]))


def test_no_2x2_level_zeros_only_coarsest_head_feature() -> None:
    model = BoardFPNCNN(**_toy_kwargs(), ablation="no_2x2_level").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["fpn_x2_energy"], torch.zeros_like(out["fpn_x2_energy"]))
    assert out["fpn_y8_energy"].abs().sum() > 0


def test_no_coordinate_planes_disables_coords() -> None:
    model = BoardFPNCNN(**_toy_kwargs(), ablation="no_coordinate_planes").eval()
    assert model.coordinates is None
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["coordinate_energy"], torch.zeros_like(out["coordinate_energy"]))


def test_bottom_up_only_zeros_topdown_updates() -> None:
    model = BoardFPNCNN(**_toy_kwargs(), ablation="bottom_up_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["topdown_4_energy"], torch.zeros_like(out["topdown_4_energy"]))
    assert torch.allclose(out["topdown_8_energy"], torch.zeros_like(out["topdown_8_energy"]))


def test_all_ablations_run_without_crash() -> None:
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    for ablation in BoardFPNCNN.ABLATIONS:
        torch.manual_seed(0)
        model = BoardFPNCNN(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation
        assert out["board_fpn_ablation"][0].item() == float(BoardFPNCNN.ABLATIONS.index(ablation))


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        BoardFPNCNN(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    model = BoardFPNCNN(**_toy_kwargs())
    bad = torch.zeros(2, 12, 8, 8)
    with pytest.raises(ValueError):
        model(bad)


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "i144"
    assert data["slug"] == "board_fpn_cnn"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "i144"
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
    assert isinstance(model, BoardFPNCNN)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.level8, BoardFPNConvStack)
    assert isinstance(model.head, BoardFPNHead)
    assert isinstance(model.coordinates, BoardFPNCoordinatePlanes)

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

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i144"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues, conformance_rows[0].issues
