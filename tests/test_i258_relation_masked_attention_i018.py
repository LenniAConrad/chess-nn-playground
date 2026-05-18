"""Focused tests for the i258 Relation-Masked Attention Graft over i018."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from chess_nn_playground.data.board_features import fen_to_simple_18
from chess_nn_playground.ideas.implementation import validate_idea_scaffold
from chess_nn_playground.ideas.implementation_kind import (
    analyze_model_wiring,
    detect_idea_implementation_kind,
)
from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.trunk.relation_masked_attention_i018 import (
    RelationMaskedAttentionGraft,
    RelationMaskedAttentionI018Net,
    build_relation_masked_attention_i018_from_config,
)


FOLDER = Path("ideas/registry/i258_relation_masked_attention_i018")


def _toy_kwargs() -> dict[str, object]:
    return {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 32,
        "hidden_dim": 32,
        "depth": 1,
        "stalk_dim": 4,
        "dropout": 0.0,
        "use_triads": True,
        "attention_enabled": True,
        "attention_num_heads": 2,
        "attention_dim": 16,
        "attention_top_k": 6,
        "attention_relation_rank": 4,
        "attention_king_boost": 0.5,
        "attention_neighborhood": "relation",
        "attention_zero_init_out": True,
        "attention_gate_init_bias": -2.0,
        "attention_dropout": 0.0,
    }


def _board_batch(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(np.stack([fen_to_simple_18(fen) for fen in fens], axis=0)).float()


def test_model_is_registered_with_expected_key() -> None:
    assert "relation_masked_attention_i018" in available_models()
    model = build_model("relation_masked_attention_i018", _toy_kwargs())
    assert isinstance(model, RelationMaskedAttentionI018Net)


def test_forward_returns_required_keys_and_shapes() -> None:
    model = RelationMaskedAttentionI018Net(**_toy_kwargs()).eval()
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",
            "8/8/8/8/8/8/7p/K6k b - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert isinstance(out, dict)
    expected = {
        "logits",
        "mechanism_energy",
        "sheaf_tension",
        "topology_pressure",
        "king_ring_pressure",
        "pin_pressure",
        "attention_entropy",
        "attention_king_share",
        "attention_gate_mean",
        "attention_delta_norm",
        "attention_neighbor_count",
        "attention_relation_bias_norm",
    }
    assert expected.issubset(out.keys())
    for key in expected:
        assert out[key].shape == (4,), key
        assert torch.isfinite(out[key]).all(), key


def test_zero_init_recovers_parent_trunk() -> None:
    """At construction the attention output is zero, so logits equal the disabled-graft logits."""

    torch.manual_seed(0)
    cfg_with = dict(_toy_kwargs())
    cfg_with["attention_enabled"] = True
    cfg_with["attention_zero_init_out"] = True
    model_with = RelationMaskedAttentionI018Net(**cfg_with).eval()

    torch.manual_seed(0)
    cfg_without = dict(_toy_kwargs())
    cfg_without["attention_enabled"] = False
    model_without = RelationMaskedAttentionI018Net(**cfg_without).eval()

    for w_name, w_param in model_without.named_parameters():
        if w_name in dict(model_with.named_parameters()):
            with torch.no_grad():
                model_with.state_dict()[w_name].copy_(w_param)

    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
        ]
    )
    with torch.inference_mode():
        logits_with = model_with(boards)["logits"]
        logits_without = model_without(boards)["logits"]

    assert torch.allclose(logits_with, logits_without, atol=1e-5), (
        logits_with - logits_without
    ).abs().max().item()


def test_force_gate_zero_disables_graft_contribution() -> None:
    cfg = dict(_toy_kwargs())
    cfg["attention_force_gate"] = 0.0
    model = RelationMaskedAttentionI018Net(**cfg).eval()
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert torch.allclose(out["attention_gate_mean"], torch.zeros(2), atol=1e-6)


@pytest.mark.parametrize("neighborhood", ["relation", "global", "king_zone", "candidate"])
def test_all_neighborhood_modes_produce_finite_logits(neighborhood: str) -> None:
    cfg = dict(_toy_kwargs())
    cfg["attention_neighborhood"] = neighborhood
    model = RelationMaskedAttentionI018Net(**cfg).eval()
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert out["logits"].shape == (3,)
    assert torch.isfinite(out["logits"]).all()


def test_scramble_relations_falsifier_runs_without_errors() -> None:
    cfg = dict(_toy_kwargs())
    cfg["scramble_relations"] = True
    model = RelationMaskedAttentionI018Net(**cfg).eval()
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
        ]
    )
    with torch.inference_mode():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()


def test_backward_gradients_reach_trunk_and_graft() -> None:
    cfg = dict(_toy_kwargs())
    cfg["attention_zero_init_out"] = False
    model = RelationMaskedAttentionI018Net(**cfg)
    boards = _board_batch(
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/P7/8/8/8/8/k1K5/8 w - - 0 1",
            "6k1/5ppp/8/8/8/8/8/3R3K w - - 0 1",
        ]
    )
    out = model(boards)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out["logits"], torch.tensor([1.0, 0.0, 1.0])
    )
    loss.backward()

    encoder_param = next(model.encoder.parameters())
    block_param = next(model.blocks[0].parameters())
    head_param = next(model.head.parameters())
    qkv_param = model.attention.qkv.weight
    out_param = model.attention.out.weight
    rel_proj_param = model.attention.rel_proj.weight

    for param in (encoder_param, block_param, head_param, qkv_param, out_param, rel_proj_param):
        assert param.grad is not None
        assert param.grad.abs().sum().item() > 0


def test_graft_unit_recovers_input_when_gate_forced_zero() -> None:
    """RelationMaskedAttentionGraft must return h unchanged when gate=0."""

    torch.manual_seed(0)
    graft = RelationMaskedAttentionGraft(d_model=16, num_heads=2, attn_dim=8, top_k=4)
    h = torch.randn(2, 64, 16)
    rel = torch.rand(2, 12, 64, 64)
    gate_override = torch.zeros(2, 1, 1)
    out, _diag = graft(h, rel, gate_override=gate_override)
    assert torch.allclose(out, h, atol=1e-6), (out - h).abs().max().item()


def test_idea_folder_validates_scaffold() -> None:
    report = validate_idea_scaffold(FOLDER)
    assert report["valid"], report


def test_idea_implementation_kind_is_bespoke() -> None:
    row = detect_idea_implementation_kind(FOLDER)
    assert row.detected_kind == "bespoke_model"
    assert row.implementation_status == "implemented"
    assert not row.issues, row.issues


def test_idea_model_py_does_not_use_research_packet_probe() -> None:
    model_py = (FOLDER / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called


def test_idea_model_py_round_trip_against_config() -> None:
    config = yaml.safe_load((FOLDER / "config.yaml").read_text(encoding="utf-8"))
    builder = build_relation_masked_attention_i018_from_config(config["model"])
    assert isinstance(builder, RelationMaskedAttentionI018Net)
    assert not isinstance(builder, ResearchPacketProbe)
