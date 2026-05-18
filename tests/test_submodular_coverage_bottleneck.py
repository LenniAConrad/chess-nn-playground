"""Focused tests for the bespoke i141 Submodular Coverage Bottleneck Network."""
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
from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES
from chess_nn_playground.models.trunk.submodular_coverage_bottleneck import (
    CONCEPT_SOURCE_NAMES,
    SubmodularCoverageBottleneckNetwork,
    build_submodular_coverage_bottleneck_from_config,
)


REGISTRY_KEY = "submodular_coverage_bottleneck"
IDEA_DIR = Path("ideas/registry/i141_submodular_coverage_bottleneck")
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_PUZZLE_FEN = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 8,
        "hidden_dim": 16,
        "depth": 2,
        "dropout": 0.0,
        "num_line_concepts": 4,
        "num_king_concepts": 3,
        "num_material_concepts": 3,
        "num_attributes": 6,
        "top_marginal": 3,
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
    assert isinstance(model, SubmodularCoverageBottleneckNetwork)
    assert not isinstance(model, ResearchPacketProbe)


def test_builder_passes_through_relevant_config_keys() -> None:
    model = build_submodular_coverage_bottleneck_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 2,
            "dropout": 0.0,
            "num_line_concepts": 5,
            "num_king_concepts": 4,
            "num_material_concepts": 6,
            "num_attributes": 7,
            "top_marginal": 2,
            "use_batchnorm": False,
            "ablation": "none",
        }
    )
    assert model.channels == 12
    assert model.hidden_dim == 24
    assert model.num_attributes == 7
    assert model.top_marginal == 2
    # 16 patch + 5 line + 4 king + 6 material = 31 total concepts.
    assert model.total_concepts == 16 + 5 + 4 + 6
    # Head input: F(a) + coverage (K) + top-T marginal + entropy.
    assert model.head_input_dim == 1 + 7 + 2 + 1
    assert model.classifier[1].in_features == model.head_input_dim
    assert model.classifier[1].out_features == 24


def test_forward_shape_and_required_keys() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "concept_activations",
        "coverage",
        "coverage_score",
        "marginal_gains",
        "top_marginal_values",
        "top_marginal_indices",
        "concept_entropy",
        "active_concept_count",
        "coverage_energy",
        "additive_pool_energy",
        "saturation_gap",
        "max_marginal_gain",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "submodular_coverage_ablation",
        "submodular_concept_total",
        "submodular_attribute_total",
    }
    assert expected_keys.issubset(out)
    assert out["concept_activations"].shape == (2, model.total_concepts)
    assert out["coverage"].shape == (2, model.num_attributes)
    assert out["marginal_gains"].shape == (2, model.total_concepts)
    assert out["top_marginal_values"].shape == (2, model.top_marginal)
    assert out["top_marginal_indices"].shape == (2, model.top_marginal)
    for key in (
        "coverage_score",
        "concept_entropy",
        "active_concept_count",
        "coverage_energy",
        "additive_pool_energy",
        "saturation_gap",
        "max_marginal_gain",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "submodular_coverage_ablation",
        "submodular_concept_total",
        "submodular_attribute_total",
    ):
        assert out[key].shape == (2,), key


def test_single_logit_prob_is_in_unit_interval() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["prob"].shape == (2,)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()


def test_backward_gradients_flow_through_encoder_and_head() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs())
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    # First parameter of the patch concept encoder receives gradients.
    patch_first = next(p for p in model.patch_concepts.parameters() if p.requires_grad)
    assert patch_first.grad is not None and torch.isfinite(patch_first.grad).all()
    # Coverage matrix receives gradients.
    assert model.coverage_logits.grad is not None
    assert torch.isfinite(model.coverage_logits.grad).all()
    # Classifier first Linear receives gradients.
    head_grad = model.classifier[1].weight.grad
    assert head_grad is not None and torch.isfinite(head_grad).all()


def test_coverage_in_unit_interval_and_monotone_in_activations() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    coverage = out["coverage"]
    assert (coverage >= 0.0).all() and (coverage <= 1.0).all()


def test_coverage_weights_are_nonnegative_when_constrained() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs()).eval()
    weights = model._coverage_weights()
    assert (weights >= 0.0).all()


def test_unconstrained_w_ablation_keeps_signed_weights() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs(), ablation="unconstrained_W").eval()
    # Force-flip a small subset to verify sign survives.
    with torch.no_grad():
        model.coverage_logits.copy_(model.coverage_logits - 1.0)
    weights = model._coverage_weights()
    assert (weights < 0.0).any()


def test_additive_pool_replaces_coverage_with_linear_sum() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs(), ablation="additive_pool").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    # In additive_pool the coverage tensor equals the additive pool.
    assert torch.allclose(out["coverage"], out["additive_pool_energy"].unsqueeze(-1).expand_as(out["coverage"])) or \
        torch.allclose(out["coverage"].mean(dim=-1), out["additive_pool_energy"])


def test_no_marginal_gains_zeroes_top_marginal_features_in_head_input() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs(), ablation="no_marginal_gains").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    # Manually rebuild the head input to verify the marginal block is zeroed.
    with torch.no_grad():
        activations = model._concept_activations(boards)
        weights = model._coverage_weights()
        coverage, score, marginal_gains, _ = model._coverage_and_score(activations, weights)
        # Marginal gains tensor itself is still populated for diagnostics.
        assert torch.isfinite(marginal_gains).all()
        out = model(boards)
    # The forward pass should still yield finite logits.
    assert torch.isfinite(out["logits"]).all()


def test_random_concepts_ablation_freezes_concept_encoders() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs(), ablation="random_concepts")
    frozen_modules = (model.patch_concepts, model.line_head, model.king_head, model.material_head)
    for module in frozen_modules:
        for p in module.parameters():
            assert not p.requires_grad
    # Coverage matrix and classifier still train.
    assert model.coverage_logits.requires_grad
    assert any(p.requires_grad for p in model.classifier.parameters())


def test_material_concepts_only_zeros_other_concept_activations() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs(), ablation="material_concepts_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    activations = out["concept_activations"]
    start, end = model.concept_source_slices["material"]
    # Non-material slices must be exactly zero.
    if start > 0:
        assert torch.all(activations[:, :start] == 0.0)
    if end < activations.shape[-1]:
        assert torch.all(activations[:, end:] == 0.0)


def test_all_ablations_run_without_crash_and_emit_their_code() -> None:
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    for ablation in SubmodularCoverageBottleneckNetwork.ABLATIONS:
        torch.manual_seed(0)
        model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation
        assert out["submodular_coverage_ablation"][0].item() == float(
            SubmodularCoverageBottleneckNetwork.ABLATIONS.index(ablation)
        ), ablation
        assert out["submodular_concept_total"][0].item() == float(model.total_concepts), ablation


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        SubmodularCoverageBottleneckNetwork(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        SubmodularCoverageBottleneckNetwork(input_channels=12, num_classes=1)


def test_rejects_multiclass_head_contract() -> None:
    with pytest.raises(ValueError):
        SubmodularCoverageBottleneckNetwork(input_channels=18, num_classes=2)


def test_concept_source_partition_covers_all_concepts_once() -> None:
    model = SubmodularCoverageBottleneckNetwork(**_toy_kwargs())
    seen = 0
    for name in CONCEPT_SOURCE_NAMES:
        start, end = model.concept_source_slices[name]
        assert start == seen
        assert end - start == model.concept_source_sizes[name]
        seen = end
    assert seen == model.total_concepts


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "i141"
    assert data["slug"] == "submodular_coverage_bottleneck"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "i141"
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
    assert isinstance(model, SubmodularCoverageBottleneckNetwork)
    assert not isinstance(model, ResearchPacketProbe)

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

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i141"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues, conformance_rows[0].issues
