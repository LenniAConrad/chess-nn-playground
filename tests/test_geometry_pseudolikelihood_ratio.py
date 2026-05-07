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
from chess_nn_playground.models.geometry_pseudolikelihood_ratio import StaticChessRelationIndex
from chess_nn_playground.models.registry import build_model
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


I036_FOLDER = Path("ideas/i036_geometry_conditioned_board_pseudo_likelihood_ratio_network")


def _load_idea_module():
    spec = importlib.util.spec_from_file_location("i036_geomplr_model", I036_FOLDER / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict:
    return yaml.safe_load((I036_FOLDER / "config.yaml").read_text(encoding="utf-8"))


def _sample_batch() -> torch.Tensor:
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/pppq1ppp/2npbn2/3Np3/2B1P3/2N2Q2/PPP2PPP/R3K2R b KQkq - 2 9",
    ]
    arrays = [torch.tensor(fen_to_tensor(fen), dtype=torch.float32) for fen in fens]
    return torch.stack(arrays, dim=0)


def test_i036_model_builds_from_config_and_forward_shape():
    config = _load_config()
    module = _load_idea_module()
    model = module.build_model_from_config(config).eval()
    x = _sample_batch()

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert output["class_logits"].shape == (2, 2)
    assert {
        "pseudo_nll_non_puzzle",
        "pseudo_nll_puzzle",
        "description_length_ratio",
        "pseudo_likelihood_ratio_logit",
        "mean_token_nll",
        "empty_token_fraction",
        "score_temperature",
    }.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key


def test_i036_registry_builder_and_optional_two_class_logits():
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


def test_i036_no_research_packet_probe_wiring_and_validates_as_bespoke():
    model_source = (I036_FOLDER / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_source
    assert "build_research_packet_probe_from_config" not in model_source

    wiring = analyze_model_wiring(I036_FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(I036_FOLDER)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(I036_FOLDER)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i036"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i036_static_relation_index_excludes_self_and_randomized_keeps_degree():
    relation_index = StaticChessRelationIndex(max_neighbors=40)
    randomized = StaticChessRelationIndex(max_neighbors=40, randomize_relations=True)

    assert relation_index.neighbor_idx.shape == (64, 40)
    for square in range(64):
        valid = relation_index.valid_neighbor_mask[square]
        neighbors = relation_index.neighbor_idx[square, valid]
        assert neighbors.numel() > 0
        assert not bool((neighbors == square).any().item())
        assert int(valid.sum().item()) == int(randomized.valid_neighbor_mask[square].sum().item())
        randomized_neighbors = randomized.neighbor_idx[square, randomized.valid_neighbor_mask[square]]
        assert not bool((randomized_neighbors == square).any().item())


def test_i036_fails_closed_for_unknown_channel_semantics():
    config = _load_config()
    model_cfg = dict(config["model"])
    model_cfg["input_channels"] = 112
    with pytest.raises(ValueError, match="verified simple_18"):
        build_model(model_cfg["name"], model_cfg)
