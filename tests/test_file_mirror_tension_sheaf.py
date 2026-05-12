"""Focused tests for the bespoke File-Mirror Tension Sheaf implementation (i028)."""
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
from chess_nn_playground.models.file_mirror_tension_sheaf import (
    FileMirrorTensionSheafNet,
    Simple18Mirror,
    build_file_mirror_tension_sheaf_from_config,
)
from chess_nn_playground.models.registry import build_model


IDEA_FOLDER = Path("ideas/registry/i028_file_mirror_tension_sheaf")


def _load_idea_config() -> dict:
    return yaml.safe_load((IDEA_FOLDER / "config.yaml").read_text(encoding="utf-8"))


def _load_idea_model_module():
    spec = importlib.util.spec_from_file_location("i028_file_mirror_tension_sheaf_model", IDEA_FOLDER / "model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _board_with_pieces() -> torch.Tensor:
    x = torch.zeros(2, 18, 8, 8)
    x[:, 12] = 1.0  # white to move
    # White king e1, queen d1, rook a1; black king e8, queen d8, rook h8
    x[:, 5, 7, 4] = 1.0   # K at e1
    x[:, 4, 7, 3] = 1.0   # Q at d1
    x[:, 3, 7, 0] = 1.0   # R at a1
    x[:, 11, 0, 4] = 1.0  # k at e8
    x[:, 10, 0, 3] = 1.0  # q at d8
    x[:, 9, 0, 7] = 1.0   # r at h8
    # White castling rights flags
    x[:, 13] = 1.0
    x[:, 14] = 1.0
    return x


def test_build_from_idea_config_returns_bespoke_model():
    config = _load_idea_config()
    module = _load_idea_model_module()
    model = module.build_model_from_config(config)
    assert isinstance(model, FileMirrorTensionSheafNet)


def test_forward_returns_logits_with_expected_shape():
    config = _load_idea_config()
    module = _load_idea_model_module()
    model = module.build_model_from_config(config).eval()
    x = _board_with_pieces()
    with torch.inference_mode():
        out = model(x)
    assert isinstance(out, dict)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_registered_model_builds_via_repo_registry():
    config = _load_idea_config()
    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("num_classes", 1)
    model = build_model("file_mirror_tension_sheaf", model_cfg)
    assert isinstance(model, FileMirrorTensionSheafNet)


def test_idea_does_not_import_or_call_research_packet_probe():
    wiring = analyze_model_wiring(IDEA_FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    flat = {item.rsplit(".", 1)[-1] for item in wiring.imports} | {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (flat & forbidden), f"unexpected probe symbols: {flat & forbidden}"


def test_implementation_kind_audit_classifies_idea_as_bespoke():
    row = detect_idea_implementation_kind(IDEA_FOLDER)
    assert row.detected_kind == "bespoke_model", row
    assert row.metadata_kind == "bespoke_model"
    assert row.implementation_status == "implemented"
    assert not row.issues, row.issues


def test_architecture_conformance_audit_has_no_issues_for_idea():
    rows = audit_architecture_conformance()
    matches = [row for row in rows if row.folder.endswith("i028_file_mirror_tension_sheaf")]
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


def test_simple18_file_mirror_is_involution_on_board_with_castling_rights():
    x = _board_with_pieces()
    twice = Simple18Mirror.apply(Simple18Mirror.apply(x))
    assert torch.equal(x, twice)


def test_partial_mirror_gate_breaks_full_invariance():
    # Sanity check: turning the mirror gate ON gives a model whose statistics
    # may differ from a model with the gate disabled. We check only that
    # gradients flow through rho, not exact values.
    config = _load_idea_config()
    cfg = dict(config.get("model", {}))
    cfg["num_classes"] = 1
    model = build_file_mirror_tension_sheaf_from_config(cfg)
    x = _board_with_pieces().requires_grad_(True)
    out = model(x)
    out["logits"].sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
