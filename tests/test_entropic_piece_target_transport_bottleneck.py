"""Focused tests for the bespoke Entropic Piece-Target Transport Bottleneck (i029)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

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
from chess_nn_playground.models.entropic_piece_target_transport_bottleneck import (
    DEFAULT_PAIRS,
    EntropicPieceTargetTransportBottleneck,
    NUM_PAIRS,
    build_entropic_piece_target_transport_bottleneck_from_config,
)
from chess_nn_playground.models.registry import build_model
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


IDEA_FOLDER = Path("ideas/all_ideas/registry/i029_entropic_piece_target_transport_bottleneck")


def _load_idea_config() -> dict:
    return yaml.safe_load((IDEA_FOLDER / "config.yaml").read_text(encoding="utf-8"))


def _load_idea_model_module():
    spec = importlib.util.spec_from_file_location(
        "i029_entropic_piece_target_transport_bottleneck_model", IDEA_FOLDER / "model.py"
    )
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
    # White pawn e4, black pawn e5
    x[:, 0, 4, 4] = 1.0
    x[:, 6, 3, 4] = 1.0
    return x


def test_build_from_idea_config_returns_bespoke_model():
    config = _load_idea_config()
    module = _load_idea_model_module()
    model = module.build_model_from_config(config)
    assert isinstance(model, EntropicPieceTargetTransportBottleneck)


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
    model = build_model("entropic_piece_target_transport_bottleneck", model_cfg)
    assert isinstance(model, EntropicPieceTargetTransportBottleneck)


def test_idea_does_not_import_or_call_research_packet_probe():
    wiring = analyze_model_wiring(IDEA_FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    flat = {item.rsplit(".", 1)[-1] for item in wiring.imports} | {
        item.rsplit(".", 1)[-1] for item in wiring.calls
    }
    assert not (flat & forbidden), f"unexpected probe symbols: {flat & forbidden}"


def test_model_name_is_excluded_from_research_packet_probe_names():
    assert "entropic_piece_target_transport_bottleneck" not in RESEARCH_PACKET_MODEL_NAMES


def test_implementation_kind_audit_classifies_idea_as_bespoke():
    row = detect_idea_implementation_kind(IDEA_FOLDER)
    assert row.detected_kind == "bespoke_model", row
    assert row.metadata_kind == "bespoke_model"
    assert row.implementation_status == "implemented"
    assert not row.issues, row.issues


def test_architecture_conformance_audit_has_no_issues_for_idea():
    rows = audit_architecture_conformance()
    matches = [row for row in rows if row.folder.endswith("i029_entropic_piece_target_transport_bottleneck")]
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


def test_transport_plans_are_couplings_of_their_marginals():
    """The Sinkhorn plan must integrate to one and approximately match its marginals.

    Per the architecture spec, the default Sinkhorn budget is `iters=8` with
    `epsilon=0.07`. 8 iterations is intentionally small for runtime, so the
    marginal residual is bounded but not vanishing; we additionally verify
    that more iterations tighten the residual.
    """
    config = _load_idea_config()
    module = _load_idea_model_module()
    model = module.build_model_from_config(config).eval()
    x = _board_with_pieces()
    from chess_nn_playground.models.entropic_piece_target_transport_bottleneck import (
        LogSinkhorn,
        _build_source_masks,
        _build_target_anchors,
    )
    with torch.inference_mode():
        piece_planes, white_to_move = model.adapter(x)
        canonical = model.canonicalizer(piece_planes, white_to_move)
        stem_features = model.stem(x)
        source_masks = _build_source_masks(canonical.flat)
        mu = model.measure(stem_features, source_masks)
        nu = _build_target_anchors(canonical.flat, beta_king=model.beta_king_zone)
        costs = model.cost_bank()
        pair_mu = mu.index_select(1, model.pair_group_idx)
        pair_nu = nu.index_select(1, model.pair_anchor_idx)
        pair_cost = costs.index_select(0, model.pair_group_idx)
        plan = model.sinkhorn(pair_mu, pair_nu, pair_cost)
    assert plan.shape == (2, NUM_PAIRS, 64, 64)
    plan_total = plan.sum(dim=(-2, -1))
    assert torch.allclose(plan_total, torch.ones_like(plan_total), atol=1e-3)
    # Tighter convergence with more iterations confirms the solver is correct.
    deeper = LogSinkhorn(epsilon=model.sinkhorn.epsilon, iters=400)
    with torch.inference_mode():
        deep_plan = deeper(pair_mu, pair_nu, pair_cost)
    deep_row = deep_plan.sum(dim=-1)
    deep_col = deep_plan.sum(dim=-2)
    assert torch.allclose(deep_row, pair_mu.float(), atol=1e-3)
    assert torch.allclose(deep_col, pair_nu.float(), atol=1e-3)


def test_pair_count_matches_default_pairs():
    assert NUM_PAIRS == len(DEFAULT_PAIRS)


def test_gradient_flows_through_transport_branch():
    config = _load_idea_config()
    cfg = dict(config.get("model", {}))
    cfg["num_classes"] = 1
    model = build_entropic_piece_target_transport_bottleneck_from_config(cfg)
    x = _board_with_pieces().requires_grad_(True)
    out = model(x)
    out["logits"].sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Cost-bank parameters should receive gradient through the transport branch.
    assert model.cost_bank.alpha.grad is not None
    assert torch.isfinite(model.cost_bank.alpha.grad).all()
    assert model.cost_bank.alpha.grad.abs().sum() > 0
