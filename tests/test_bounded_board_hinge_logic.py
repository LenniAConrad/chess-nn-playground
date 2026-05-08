from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind
from chess_nn_playground.models.bounded_board_hinge_logic import BoundedBoardHingeLogicNet
from chess_nn_playground.models.registry import build_model


IDEA_FOLDER = Path("ideas/i089_bounded_board_hinge_logic")


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("i089_model", IDEA_FOLDER / "model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict:
    return yaml.safe_load((IDEA_FOLDER / "config.yaml").read_text(encoding="utf-8"))


def _make_board(batch: int = 2, channels: int = 18) -> torch.Tensor:
    x = torch.zeros(batch, channels, 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    return x


def test_idea_local_wrapper_has_no_research_packet_probe():
    text = (IDEA_FOLDER / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in text
    assert "build_research_packet_probe_from_config" not in text


def test_wrapper_builds_bespoke_model_from_config():
    module = _load_wrapper()
    config = _load_config()
    model = module.build_model_from_config(config)
    assert isinstance(model, BoundedBoardHingeLogicNet)


def test_registry_resolves_bespoke_model():
    config = _load_config()
    model = build_model("bounded_board_hinge_logic", config["model"])
    assert isinstance(model, BoundedBoardHingeLogicNet)


def test_forward_pass_returns_one_logit_per_board():
    config = _load_config()
    module = _load_wrapper()
    model = module.build_model_from_config(config).eval()
    x = _make_board()
    with torch.inference_mode():
        output = model(x)
    assert isinstance(output, dict)
    logits = output["logits"]
    assert logits.shape == (x.shape[0],)
    assert torch.isfinite(logits).all()
    truths = output["formula_truths"]
    assert truths.shape == (x.shape[0], 24 + 96 + 48)
    assert torch.all(truths >= 0.0) and torch.all(truths <= 1.0)
    assert "logic_energy_gap" in output
    assert "psl_energy_y0" in output
    assert "psl_energy_y1" in output


def test_implementation_kind_detected_as_bespoke():
    row = detect_idea_implementation_kind(IDEA_FOLDER)
    assert row.detected_kind == "bespoke_model"
    assert row.metadata_kind == "bespoke_model"
    assert row.implementation_status == "implemented"
    assert not row.issues


def test_architecture_conformance_passes_for_idea():
    rows = audit_architecture_conformance()
    matches = [row for row in rows if row.idea_id == "i089"]
    assert matches, "i089 must appear in architecture conformance rows once implemented"
    assert all(not row.issues for row in matches), matches
    assert all(row.implementation_kind == "bespoke_model" for row in matches)
    assert all(row.architecture_has_binding_section for row in matches)
    assert all(row.architecture_mentions_model_name for row in matches)
    assert all(row.architecture_mentions_source for row in matches)
    assert all(row.architecture_mentions_wrapper for row in matches)


@pytest.mark.parametrize("input_channels", [18])
def test_forward_pass_input_channels(input_channels: int):
    model = BoundedBoardHingeLogicNet(input_channels=input_channels).eval()
    x = _make_board(batch=3, channels=input_channels)
    with torch.inference_mode():
        output = model(x)
    assert output["logits"].shape == (3,)
