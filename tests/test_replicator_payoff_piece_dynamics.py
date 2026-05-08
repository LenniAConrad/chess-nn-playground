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
from chess_nn_playground.models.registry import build_model
from chess_nn_playground.models.replicator_payoff_piece_dynamics import (
    OccupiedPieceTokenizer,
    ReplicatorPayoffPieceDynamicsNetwork,
)
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


I131_FOLDER = Path("ideas/i131_replicator_payoff_piece_dynamics")


def _load_idea_module():
    spec = importlib.util.spec_from_file_location("i131_model", I131_FOLDER / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict:
    return yaml.safe_load((I131_FOLDER / "config.yaml").read_text(encoding="utf-8"))


def _sample_batch() -> torch.Tensor:
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/pppq1ppp/2npbn2/3Np3/2B1P3/2N2Q2/PPP2PPP/R3K2R b KQkq - 2 9",
    ]
    arrays = [torch.tensor(fen_to_tensor(fen), dtype=torch.float32) for fen in fens]
    return torch.stack(arrays, dim=0)


def test_i131_model_builds_from_config_and_forward_shape():
    config = _load_config()
    module = _load_idea_module()
    model = module.build_model_from_config(config).eval()
    x = _sample_batch()

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "mean_entropy",
        "mean_top_mass",
        "mean_kl_from_initial",
        "mean_avg_payoff",
        "mean_fitness_variance",
        "payoff_asymmetry_norm",
        "total_piece_count",
        "backbone_feature_norm",
        "head0_entropy",
        "head0_top_mass",
        "head0_kl_from_initial",
        "head0_avg_payoff",
        "head0_fitness_variance",
        "head0_own_mass",
        "head0_opp_mass",
        "head0_king_mass",
        "head0_pawn_mass",
        "head0_minor_mass",
        "head0_major_mass",
    }
    assert expected_keys.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key
    # Starting position has 32 pieces; the middle-game FEN has 29 pieces.
    assert torch.allclose(output["total_piece_count"], torch.tensor([32.0, 29.0]))
    # King mass over heads should reflect that two kings are present (own+opp).
    assert (output["head0_king_mass"] >= 0).all()
    assert (output["head0_king_mass"] <= 1).all()


def test_i131_registry_builder_matches_idea_wrapper():
    config = _load_config()
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = build_model(model_name, model_cfg).eval()
    x = _sample_batch()

    with torch.no_grad():
        output = model(x)

    assert isinstance(model, ReplicatorPayoffPieceDynamicsNetwork)
    assert output["logits"].shape == (2,)
    assert model_name == "replicator_payoff_piece_dynamics"
    # The promotion to a bespoke model removes the slug from the
    # research-packet probe roster.
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES


def test_i131_replicator_population_stays_normalised_and_masked():
    torch.manual_seed(0)
    network = ReplicatorPayoffPieceDynamicsNetwork(
        channels=16,
        hidden_dim=24,
        depth=1,
        max_pieces=16,
        token_dim=16,
        pair_hidden_dim=16,
        num_heads=2,
        num_steps=3,
    ).eval()

    x = torch.zeros(2, 18, 8, 8)
    x[:, 12] = 1.0
    # Place a few pieces.
    x[:, 5, 7, 4] = 1.0  # own king
    x[:, 11, 0, 4] = 1.0  # opp king
    x[:, 0, 6, 4] = 1.0  # own pawn
    x[:, 6, 1, 4] = 1.0  # opp pawn
    x[:, 3, 7, 0] = 1.0  # own rook
    x[:, 9, 0, 0] = 1.0  # opp rook

    tokenizer = network.tokenizer
    token_raw, mask, pair_geometry, sel_idx = tokenizer(x)
    assert token_raw.shape == (2, tokenizer.max_pieces, tokenizer.raw_dim)
    assert mask.shape == (2, tokenizer.max_pieces)
    assert pair_geometry.shape == (
        2,
        tokenizer.max_pieces,
        tokenizer.max_pieces,
        tokenizer.geometry_dim,
    )
    assert sel_idx.shape == (2, tokenizer.max_pieces)
    assert mask.sum(dim=-1).tolist() == [6.0, 6.0]
    assert ((mask == 0) | (mask == 1)).all()

    tokens = network.token_proj(token_raw)
    pmax = tokenizer.max_pieces
    ti = tokens.unsqueeze(2).expand(2, pmax, pmax, tokens.shape[-1])
    tj = tokens.unsqueeze(1).expand(2, pmax, pmax, tokens.shape[-1])
    payoff = network.payoff_mlp(torch.cat([ti, tj, pair_geometry], dim=-1))
    payoff = payoff.permute(0, 3, 1, 2).contiguous()
    init_scores = network.init_logits(tokens).permute(0, 2, 1)
    mask_expand = mask.unsqueeze(1).expand(-1, network.num_heads, -1)
    final_p, initial_p, _, avg_history = network._replicator(payoff, mask_expand, init_scores)

    # Population renormalises every step.
    assert torch.allclose(initial_p.sum(dim=-1), torch.ones(2, network.num_heads), atol=1e-5)
    assert torch.allclose(final_p.sum(dim=-1), torch.ones(2, network.num_heads), atol=1e-5)
    # Padded slots receive negligible mass.
    padded_mass = (final_p * (1.0 - mask_expand)).sum(dim=-1)
    assert (padded_mass < 1e-5).all()
    assert len(avg_history) == network.num_steps


def test_i131_no_research_packet_probe_wiring_and_validates_as_bespoke():
    model_source = (I131_FOLDER / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_source
    assert "build_research_packet_probe_from_config" not in model_source

    wiring = analyze_model_wiring(I131_FOLDER / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(I131_FOLDER)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert kind_row.idea_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(I131_FOLDER)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i131"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i131_rejects_multi_class_head():
    with pytest.raises(ValueError, match="puzzle_binary one-logit"):
        ReplicatorPayoffPieceDynamicsNetwork(num_classes=2)


def test_i131_tokenizer_geometry_table_marks_knight_and_diagonals():
    tokenizer = OccupiedPieceTokenizer(max_pieces=4)
    geom = tokenizer.geometry  # (64, 64, G)
    # Square (rank=0,file=0) -> index 0; knight square (rank=2,file=1) -> index 2*8+1=17
    knight_features = geom[0, 17]
    assert knight_features[-1].item() == pytest.approx(1.0)
    # Diagonal a1 -> h8 (index 63)
    diag = geom[0, 63]
    assert diag[6].item() == pytest.approx(1.0)
    # Same-file a1 -> a8 (index 56)
    same_file = geom[0, 56]
    assert same_file[4].item() == pytest.approx(1.0)
