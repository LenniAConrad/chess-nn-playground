from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path

import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_tensor
from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.implementation_kind import analyze_model_wiring
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind
from chess_nn_playground.models.trunk.mobius_piece_constellation import ElementarySymmetricInteractionBlock
from chess_nn_playground.models.registry import build_model
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


I037_FOLDER = Path("ideas/registry/i037_mobius_piece_constellation_network")


def _load_idea_module():
    spec = importlib.util.spec_from_file_location("i037_mobius_model", I037_FOLDER / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict:
    return yaml.safe_load((I037_FOLDER / "config.yaml").read_text(encoding="utf-8"))


def _sample_batch() -> torch.Tensor:
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/pppq1ppp/2npbn2/3Np3/2B1P3/2N2Q2/PPP2PPP/R3K2R b KQkq - 2 9",
    ]
    arrays = [torch.tensor(fen_to_tensor(fen), dtype=torch.float32) for fen in fens]
    return torch.stack(arrays, dim=0)


def test_i037_model_builds_from_config_and_forward_shape():
    config = _load_config()
    module = _load_idea_module()
    model = module.build_model_from_config(config).eval()
    x = _sample_batch()

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert {
        "degree1_norm",
        "degree2_norm",
        "degree3_norm",
        "occupied_count",
        "mean_occupancy",
        "degree_gate_mean",
        "degree_gate_l1",
        "state_embedding_norm",
    }.issubset(output)
    assert output["degree_gate_mean"].shape == (2, 3)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key


def test_i037_registry_builder_and_optional_two_class_logits():
    config = _load_config()
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, model_cfg).eval()
    x = _sample_batch()

    with torch.no_grad():
        output = model(x)

    assert output["logits"].shape == (2,)
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    two_class_cfg = dict(model_cfg)
    two_class_cfg["num_classes"] = 2
    two_class_model = build_model(model_name, two_class_cfg).eval()
    with torch.no_grad():
        two_class_output = two_class_model(x)
    assert two_class_output["logits"].shape == (2, 2)


def test_i037_elementary_symmetric_recurrence_matches_explicit_enumeration():
    tokens = torch.tensor(
        [
            [
                [1.0, 2.0, -1.0],
                [0.5, -1.0, 3.0],
                [-2.0, 0.25, 0.5],
                [1.5, 1.0, 2.0],
            ]
        ]
    )
    occupancy = torch.ones(1, 4)
    block = ElementarySymmetricInteractionBlock(max_degree=3, normalize_by_tuple_count=False)

    h1, h2, h3 = block(tokens, occupancy)

    explicit_h1 = tokens.sum(dim=1)
    explicit_h2 = torch.stack(
        [sum(tokens[0, i] * tokens[0, j] for i, j in combinations(range(4), 2))]
    )
    explicit_h3 = torch.stack(
        [sum(tokens[0, i] * tokens[0, j] * tokens[0, k] for i, j, k in combinations(range(4), 3))]
    )
    assert torch.allclose(h1, explicit_h1)
    assert torch.allclose(h2, explicit_h2)
    assert torch.allclose(h3, explicit_h3)


def test_i037_no_research_packet_probe_wiring_and_validates_as_bespoke():
    model_source = (I037_FOLDER / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_source
    assert "build_research_packet_probe_from_config" not in model_source

    wiring = analyze_model_wiring(I037_FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(I037_FOLDER)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(I037_FOLDER)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i037"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i037_fails_closed_without_verified_channel_map():
    config = _load_config()
    model_cfg = dict(config["model"])
    model_cfg["input_channels"] = 112
    model_cfg["encoding"] = "lc0_static_112"
    model_cfg["channel_map"] = None
    with pytest.raises(ValueError, match="explicit current-board piece-plane channel_map"):
        build_model(model_cfg["name"], model_cfg)
