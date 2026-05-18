"""Focused tests for the bespoke i241 Multi-Stream Attention Chess Evaluator."""
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
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES
from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.trunk.multistream_attention_chess_eval import (
    MultistreamAttentionChessEval,
    MultistreamTransformerBlock,
    StreamProjection,
    build_multistream_attention_chess_eval_from_config,
)


REGISTRY_KEY = "multistream_attention_chess_eval"
IDEA_DIR = Path("ideas/registry/i241_multistream_attention_chess_eval")
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_PUZZLE_FEN = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "embed_dim": 16,
        "num_heads": 2,
        "exchange_blocks": 1,
        "king_blocks": 1,
        "positional_blocks": 1,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "aux_loss_weight": 0.05,
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
    assert isinstance(model, MultistreamAttentionChessEval)
    assert not isinstance(model, ResearchPacketProbe)


def test_builder_passes_through_channels_alias() -> None:
    cfg = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 16,
        "num_heads": 2,
        "exchange_blocks": 1,
        "king_blocks": 1,
        "positional_blocks": 1,
        "dropout": 0.0,
    }
    model = build_multistream_attention_chess_eval_from_config(cfg)
    assert model.embed_dim == 16


def test_forward_shape_and_required_keys() -> None:
    model = MultistreamAttentionChessEval(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "exchange_logit",
        "king_logit",
        "positional_logit",
        "alpha_exchange",
        "alpha_king",
        "alpha_positional",
        "residual_logit",
        "route_entropy",
        "stream_disagreement",
        "exchange_pool_norm",
        "king_pool_norm",
        "positional_pool_norm",
        "exchange_aux_logit",
        "king_aux_logit",
        "positional_aux_logit",
        "aux_loss_weight",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "multistream_ablation",
        "multistream_stream_count",
    }
    assert expected_keys.issubset(out)
    for key in expected_keys - {"logits", "prob"}:
        assert out[key].shape[0] == 2, key


def test_alpha_softmax_sums_to_one() -> None:
    model = MultistreamAttentionChessEval(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    summed = out["alpha_exchange"] + out["alpha_king"] + out["alpha_positional"]
    assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)


def test_prob_is_in_unit_interval() -> None:
    model = MultistreamAttentionChessEval(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()


def test_no_phase_router_returns_uniform_mixture() -> None:
    model = MultistreamAttentionChessEval(**_toy_kwargs(), ablation="no_phase_router").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    expected = torch.full_like(out["alpha_exchange"], 1.0 / 3.0)
    assert torch.allclose(out["alpha_exchange"], expected)
    assert torch.allclose(out["alpha_king"], expected)
    assert torch.allclose(out["alpha_positional"], expected)


def test_remove_streams_zero_pool_norms() -> None:
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    for ablation, expected_zero in (
        ("remove_exchange_stream", "exchange_pool_norm"),
        ("remove_king_stream", "king_pool_norm"),
        ("remove_positional_stream", "positional_pool_norm"),
    ):
        model = MultistreamAttentionChessEval(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.allclose(
            out[expected_zero], torch.zeros_like(out[expected_zero])
        ), ablation


def test_no_aux_heads_zero_aux_logits() -> None:
    model = MultistreamAttentionChessEval(**_toy_kwargs(), ablation="no_aux_heads").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["exchange_aux_logit"], torch.zeros_like(out["exchange_aux_logit"]))
    assert torch.allclose(out["king_aux_logit"], torch.zeros_like(out["king_aux_logit"]))
    assert torch.allclose(out["positional_aux_logit"], torch.zeros_like(out["positional_aux_logit"]))


def test_backward_gradients_flow_through_streams_and_head() -> None:
    model = MultistreamAttentionChessEval(**_toy_kwargs())
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    has_exchange_grad = any(
        (p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in model.exchange_blocks[0].parameters()
    )
    has_router_grad = any(
        (p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in model.phase_router.parameters()
    )
    assert has_exchange_grad
    assert has_router_grad


def test_all_ablations_run_without_crash() -> None:
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    for ablation in MultistreamAttentionChessEval.ABLATIONS:
        torch.manual_seed(0)
        model = MultistreamAttentionChessEval(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation
        assert out["multistream_ablation"][0].item() == float(
            MultistreamAttentionChessEval.ABLATIONS.index(ablation)
        )


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        MultistreamAttentionChessEval(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        MultistreamAttentionChessEval(input_channels=12, num_classes=1)


def test_rejects_non_binary_classes() -> None:
    with pytest.raises(ValueError):
        MultistreamAttentionChessEval(num_classes=3)


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "i241"
    assert data["slug"] == "multistream_attention_chess_eval"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "i241"
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
    assert isinstance(model, MultistreamAttentionChessEval)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.exchange_proj, StreamProjection)
    assert all(isinstance(block, MultistreamTransformerBlock) for block in model.exchange_blocks)

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

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i241"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues, conformance_rows[0].issues
