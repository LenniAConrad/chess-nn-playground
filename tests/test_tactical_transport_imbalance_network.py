from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml

from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation_kind import analyze_model_wiring
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind
from chess_nn_playground.models.registry import build_model


IDEA_FOLDER = Path("ideas/all_ideas/registry/i031_tactical_transport_imbalance_network")


def _load_config() -> dict:
    return yaml.safe_load((IDEA_FOLDER / "config.yaml").read_text(encoding="utf-8"))


def _load_idea_model_module():
    spec = importlib.util.spec_from_file_location("i031_model", IDEA_FOLDER / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_simple18_batch() -> torch.Tensor:
    x = torch.zeros(2, 18, 8, 8)
    x[0, 12] = 1.0
    x[0, 0, 6, 4] = 1.0
    x[0, 3, 7, 0] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 10, 1, 4] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[1, 12] = 0.0
    x[1, 6, 1, 4] = 1.0
    x[1, 9, 0, 0] = 1.0
    x[1, 11, 0, 4] = 1.0
    x[1, 4, 6, 4] = 1.0
    x[1, 5, 7, 4] = 1.0
    return x


def test_i031_builds_from_idea_config_and_forward_shape() -> None:
    config = _load_config()
    module = _load_idea_model_module()
    model = module.build_model_from_config(config).eval()

    with torch.no_grad():
        output = model(_synthetic_simple18_batch())

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key in {
        "transport_imbalance",
        "forward_transport_cost",
        "reverse_transport_cost",
        "transport_entropy_gap",
    }:
        assert output[key].shape == (2,)
        assert torch.isfinite(output[key]).all()


def test_i031_registry_builder_matches_config_contract() -> None:
    config = _load_config()
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, model_cfg).eval()

    with torch.no_grad():
        output = model(_synthetic_simple18_batch())

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)


def test_i031_wrapper_does_not_use_research_packet_probe() -> None:
    wiring = analyze_model_wiring(IDEA_FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}

    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called


def test_i031_is_bespoke_and_architecture_conformant() -> None:
    kind_row = detect_idea_implementation_kind(IDEA_FOLDER)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    rows = [row for row in audit_architecture_conformance() if row.idea_id == "i031"]
    assert len(rows) == 1
    assert rows[0].implementation_kind == "bespoke_model"
    assert not rows[0].issues
