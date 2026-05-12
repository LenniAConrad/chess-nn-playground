"""Focused tests for the bespoke Nuisance-Orthogonal Puzzle Bottleneck (i030)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation_kind import analyze_model_wiring
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind
from chess_nn_playground.models.nuisance_orthogonal_puzzle_bottleneck import (
    BatchRidgeOrthogonalProjector,
    DeterministicNuisanceExtractor,
    FixedNuisanceFeatureMap,
    NUISANCE_BASE_DIM,
    NuisanceOrthogonalPuzzleNet,
    Simple18Adapter,
    build_nuisance_orthogonal_puzzle_bottleneck_from_config,
)
from chess_nn_playground.models.registry import build_model
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


IDEA_FOLDER = Path("ideas/all_ideas/registry/i030_nuisance_orthogonal_puzzle_bottleneck")


def _load_idea_config() -> dict:
    return yaml.safe_load((IDEA_FOLDER / "config.yaml").read_text(encoding="utf-8"))


def _load_idea_model_module():
    spec = importlib.util.spec_from_file_location(
        "i030_nuisance_orthogonal_puzzle_bottleneck_model", IDEA_FOLDER / "model.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _board_batch(batch: int = 8) -> torch.Tensor:
    """Return a deterministic but varied batch of simple_18 boards."""
    torch.manual_seed(0)
    x = torch.zeros(batch, 18, 8, 8)
    for i in range(batch):
        # Side to move alternates so the side-to-move scalar varies.
        if i % 2 == 0:
            x[i, 12] = 1.0
        # Always place both kings so king-coordinate features are well defined.
        x[i, 5, 7, (i + 4) % 8] = 1.0   # K (white) on rank 1
        x[i, 11, 0, (i + 3) % 8] = 1.0  # k (black) on rank 8
        # A few pawns for material variation.
        for f in range(min(8, 4 + i % 5)):
            x[i, 0, 6, f] = 1.0
            x[i, 6, 1, f] = 1.0
        # A castling flag toggled per index so the nuisance vector varies.
        x[i, 13 + (i % 4)] = 1.0
        # En-passant on a varying file.
        x[i, 17, 5, i % 8] = 1.0
    return x


def test_build_from_idea_config_returns_bespoke_model():
    config = _load_idea_config()
    module = _load_idea_model_module()
    model = module.build_model_from_config(config)
    assert isinstance(model, NuisanceOrthogonalPuzzleNet)


def test_forward_returns_logits_with_expected_shape():
    config = _load_idea_config()
    module = _load_idea_model_module()
    model = module.build_model_from_config(config).eval()
    x = _board_batch(batch=8)
    with torch.inference_mode():
        out = model(x)
    assert isinstance(out, dict)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()


def test_smoke_test_compatible_two_sample_batch():
    """Mirrors the project-wide smoke test contract: batch=2 must produce shape (2,)."""
    config = _load_idea_config()
    module = _load_idea_model_module()
    model = module.build_model_from_config(config).eval()
    x = _board_batch(batch=2)
    with torch.inference_mode():
        out = model(x)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()


def test_registered_model_builds_via_repo_registry():
    config = _load_idea_config()
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    model = build_model("nuisance_orthogonal_puzzle_bottleneck", model_cfg)
    assert isinstance(model, NuisanceOrthogonalPuzzleNet)


def test_idea_does_not_import_or_call_research_packet_probe():
    wiring = analyze_model_wiring(IDEA_FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    flat = {item.rsplit(".", 1)[-1] for item in wiring.imports} | {
        item.rsplit(".", 1)[-1] for item in wiring.calls
    }
    assert not (flat & forbidden), f"unexpected probe symbols: {flat & forbidden}"

    model_py = (IDEA_FOLDER / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py


def test_model_name_is_excluded_from_research_packet_probe_names():
    assert "nuisance_orthogonal_puzzle_bottleneck" not in RESEARCH_PACKET_MODEL_NAMES


def test_implementation_kind_audit_classifies_idea_as_bespoke():
    row = detect_idea_implementation_kind(IDEA_FOLDER)
    assert row.detected_kind == "bespoke_model", row
    assert row.metadata_kind == "bespoke_model"
    assert row.implementation_status == "implemented"
    assert not row.issues, row.issues


def test_architecture_conformance_audit_has_no_issues_for_idea():
    rows = audit_architecture_conformance()
    matches = [row for row in rows if row.folder.endswith("i030_nuisance_orthogonal_puzzle_bottleneck")]
    assert len(matches) == 1
    row = matches[0]
    assert row.implementation_kind == "bespoke_model"
    assert row.architecture_has_binding_section
    assert row.architecture_mentions_model_name
    assert row.architecture_mentions_source
    assert row.architecture_mentions_wrapper
    assert row.source_files
    assert not row.source_markers
    assert not row.issues, row.issues


def test_nuisance_extractor_dimension_and_finiteness():
    adapter = Simple18Adapter(input_channels=18, encoding="simple_18")
    extractor = DeterministicNuisanceExtractor()
    x = _board_batch(batch=4)
    n = extractor(adapter(x))
    assert n.shape == (4, NUISANCE_BASE_DIM)
    assert torch.isfinite(n).all()


def test_simple18_adapter_fails_closed_on_unknown_encoding():
    with pytest.raises(ValueError):
        Simple18Adapter(input_channels=18, encoding="lc0_static_112")
    with pytest.raises(ValueError):
        Simple18Adapter(input_channels=20, encoding="simple_18")


def test_projection_residual_orthogonality_with_zero_ridge():
    """When lambda=0 and rank(Q)=k the projection enforces Q^T Z = 0 exactly."""
    torch.manual_seed(7)
    b, d, k = 32, 24, 8
    h = torch.randn(b, d)
    q = torch.randn(b, k)
    projector = BatchRidgeOrthogonalProjector(ridge_lambda=0.0, gamma=1.0)
    out = projector(h, q)
    z = out["z"].float()
    q_centred = (q - q.mean(dim=0, keepdim=True)).float()
    residual = q_centred.transpose(0, 1) @ z
    assert residual.abs().max() < 1e-3


def test_projection_residual_orthogonality_with_ridge():
    """With lambda>0 the residual covariance shrinks toward zero with growing batch."""
    torch.manual_seed(11)
    b, d, k = 64, 16, 6
    h = torch.randn(b, d)
    q = torch.randn(b, k)
    projector = BatchRidgeOrthogonalProjector(ridge_lambda=1e-6, gamma=1.0)
    out = projector(h, q)
    cov = out["residual_cov_norm"].item()
    assert cov < 1e-3, cov


def test_projection_gamma_zero_returns_centred_latent():
    torch.manual_seed(13)
    b, d, k = 16, 8, 4
    h = torch.randn(b, d)
    q = torch.randn(b, k)
    projector = BatchRidgeOrthogonalProjector(ridge_lambda=1e-3, gamma=0.0)
    out = projector(h, q)
    z = out["z"].float()
    expected = h - h.mean(dim=0, keepdim=True)
    assert torch.allclose(z, expected, atol=1e-5)


def test_fixed_nuisance_feature_map_is_deterministic_and_non_trainable():
    fm_a = FixedNuisanceFeatureMap(in_dim=NUISANCE_BASE_DIM, rank=32, expansion_dim=48, seed=42)
    fm_b = FixedNuisanceFeatureMap(in_dim=NUISANCE_BASE_DIM, rank=32, expansion_dim=48, seed=42)
    n = torch.randn(8, NUISANCE_BASE_DIM)
    qa = fm_a(n)
    qb = fm_b(n)
    assert torch.allclose(qa, qb)
    assert qa.shape == (8, 32)
    # Random projection lives in a buffer, so the only learned parameter is None
    # (LayerNorm has elementwise_affine=False so it has no parameters either).
    trainable_params = [p for p in fm_a.parameters() if p.requires_grad]
    assert trainable_params == []


def test_gradient_flows_through_trunk_and_head():
    config = _load_idea_config()
    cfg = dict(config["model"])
    cfg["num_classes"] = 1
    model = build_nuisance_orthogonal_puzzle_bottleneck_from_config(cfg)
    x = _board_batch(batch=8).requires_grad_(True)
    out = model(x)
    out["logits"].sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Trunk and head parameters must receive gradient.
    trunk_grads = [p.grad for p in model.trunk.parameters() if p.requires_grad]
    head_grads = [p.grad for p in model.head.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in trunk_grads)
    assert all(g is not None and torch.isfinite(g).all() for g in head_grads)
    assert sum(g.abs().sum().item() for g in trunk_grads) > 0
    assert sum(g.abs().sum().item() for g in head_grads) > 0
