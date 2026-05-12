from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_tensor
from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.implementation_kind import analyze_model_wiring
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind
from chess_nn_playground.models.registry import build_model
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


I038_FOLDER = Path("ideas/registry/i038_sparse_witness_piece_bottleneck_network")


def _load_idea_module():
    spec = importlib.util.spec_from_file_location("i038_sparse_witness_model", I038_FOLDER / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict:
    return yaml.safe_load((I038_FOLDER / "config.yaml").read_text(encoding="utf-8"))


def _sample_batch() -> torch.Tensor:
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/8/8/3k4/8/3Q4/8/4K3 w - - 0 1",
    ]
    arrays = [torch.tensor(fen_to_tensor(fen), dtype=torch.float32) for fen in fens]
    return torch.stack(arrays, dim=0)


def _occupancy(x: torch.Tensor) -> torch.Tensor:
    return x[:, :12].sum(dim=1, keepdim=True).clamp(0.0, 1.0)


def test_i038_model_builds_from_config_and_forward_shape():
    config = _load_config()
    module = _load_idea_module()
    model = module.build_model_from_config(config).eval()
    x = _sample_batch()

    with torch.no_grad():
        logits = model(x)
        debug_logits, mask, raw_scores = model.forward_with_mask(x)

    assert logits.shape == (2,)
    assert torch.allclose(logits, debug_logits)
    assert mask.shape == (2, 1, 8, 8)
    assert raw_scores.shape == (2, 1, 8, 8)
    assert torch.isfinite(logits).all()


def test_i038_witness_mask_is_occupied_only_and_budgeted():
    config = _load_config()
    model = _load_idea_module().build_model_from_config(config).eval()
    x = _sample_batch()
    occupied = _occupancy(x)
    budget = int(config["model"]["witness_budget"])

    with torch.no_grad():
        _logits, mask, raw_scores = model.forward_with_mask(x)

    assert torch.all((mask == 0.0) | (mask == 1.0))
    assert torch.all(mask <= occupied)
    expected_counts = torch.minimum(
        occupied.flatten(1).sum(dim=1),
        torch.full((x.shape[0],), float(budget)),
    )
    assert torch.allclose(mask.flatten(1).sum(dim=1), expected_counts)
    assert (raw_scores[occupied <= 0.5] < -1.0e8).all()

    with torch.no_grad():
        _logits2, mask2, _raw_scores2 = model.forward_with_mask(x)
    assert torch.equal(mask, mask2)


def test_i038_registry_builder_and_optional_two_class_logits():
    config = _load_config()
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, model_cfg).eval()
    x = _sample_batch()

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2,)
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    two_class_cfg = dict(model_cfg)
    two_class_cfg["num_classes"] = 2
    two_class_model = build_model(model_name, two_class_cfg).eval()
    with torch.no_grad():
        two_class_logits = two_class_model(x)
    assert two_class_logits.shape == (2, 2)


def test_i038_no_research_packet_probe_wiring_and_validates_as_bespoke():
    model_source = (I038_FOLDER / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_source
    assert "build_research_packet_probe_from_config" not in model_source

    wiring = analyze_model_wiring(I038_FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(I038_FOLDER)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(I038_FOLDER)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i038"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i038_fails_closed_without_verified_channel_map():
    config = _load_config()
    model_cfg = dict(config["model"])
    model_cfg["input_channels"] = 112
    model_cfg["encoding"] = "lc0_static_112"
    model_cfg["adapter"] = None
    with pytest.raises(ValueError, match="requires exactly 12 current piece_plane_indices"):
        build_model(model_cfg["name"], model_cfg)
