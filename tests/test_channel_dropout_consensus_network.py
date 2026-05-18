"""Focused tests for the bespoke i118 Channel Dropout Consensus Network."""
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
from chess_nn_playground.models.trunk.channel_dropout_consensus import (
    DETERMINISTIC_VIEW_DROP_CHANNELS,
    DETERMINISTIC_VIEW_NAMES,
    ChannelDropoutConsensusNetwork,
    _SharedBoardEncoder,
    build_channel_dropout_consensus_network_from_config,
)


REGISTRY_KEY = "channel_dropout_consensus_network"
IDEA_DIR = Path("ideas/registry/i118_channel_dropout_consensus_network")
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_PUZZLE_FEN = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "channels": 8,
        "hidden_dim": 32,
        "depth": 2,
        "dropout": 0.0,
        "view_dropout_p": 0.5,
        "random_mask_seed": 7,
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
    assert isinstance(model, ChannelDropoutConsensusNetwork)
    assert not isinstance(model, ResearchPacketProbe)


def test_builder_passes_through_relevant_config_keys() -> None:
    model = build_channel_dropout_consensus_network_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 12,
            "hidden_dim": 24,
            "depth": 1,
            "dropout": 0.0,
            "view_dropout_p": 0.25,
            "random_mask_seed": 99,
            "use_batchnorm": False,
            "ablation": "none",
        }
    )
    assert model.channels == 12
    assert model.hidden_dim == 24
    assert model.depth == 1
    assert model.view_dropout_p == 0.25
    assert model.random_mask_seed == 99
    # The head's first Linear receives 4 * channels features.
    assert model.head_input_dim == 4 * 12
    # Head structure: LayerNorm, Linear(in -> hidden), GELU, (optional dropout), Linear(hidden -> 1)
    assert model.classifier[1].in_features == 4 * 12
    assert model.classifier[1].out_features == 24


def test_forward_shape_and_required_keys() -> None:
    model = ChannelDropoutConsensusNetwork(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "view_pooled",
        "mean_latent",
        "variance_latent",
        "max_pairwise",
        "full_view_latent",
        "consensus_energy",
        "disagreement_energy",
        "max_pairwise_energy",
        "full_view_energy",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "channel_dropout_ablation",
        "channel_dropout_view_count",
    }
    assert expected_keys.issubset(out)
    assert out["view_pooled"].shape == (2, model.num_views, model.channels)
    for key in ("mean_latent", "variance_latent", "max_pairwise", "full_view_latent"):
        assert out[key].shape == (2, model.channels), key
    for key in (
        "consensus_energy",
        "disagreement_energy",
        "max_pairwise_energy",
        "full_view_energy",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "channel_dropout_ablation",
        "channel_dropout_view_count",
    ):
        assert out[key].shape == (2,), key


def test_single_logit_prob_is_in_unit_interval() -> None:
    model = ChannelDropoutConsensusNetwork(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["prob"].shape == (2,)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()


def test_backward_gradients_flow_through_encoder_and_head() -> None:
    model = ChannelDropoutConsensusNetwork(**_toy_kwargs())
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    encoder_first_conv = next(p for p in model.encoder.parameters() if p.requires_grad)
    assert encoder_first_conv.grad is not None and torch.isfinite(encoder_first_conv.grad).all()
    head_first = model.classifier[1].weight.grad
    assert head_first is not None and torch.isfinite(head_first).all()


def test_view_masks_are_buffers_not_parameters() -> None:
    model = ChannelDropoutConsensusNetwork(**_toy_kwargs())
    parameter_ids = {id(p) for p in model.parameters()}
    assert id(model.view_masks) not in parameter_ids


def test_view_masks_match_deterministic_drop_groups() -> None:
    model = ChannelDropoutConsensusNetwork(**_toy_kwargs()).eval()
    assert model.view_masks.shape == (len(DETERMINISTIC_VIEW_NAMES), 18)
    for view_idx, view_name in enumerate(DETERMINISTIC_VIEW_NAMES):
        drop_channels = set(DETERMINISTIC_VIEW_DROP_CHANNELS[view_name])
        for ch in range(18):
            if ch in drop_channels:
                assert model.view_masks[view_idx, ch].item() == 0.0, (view_name, ch)
            else:
                assert model.view_masks[view_idx, ch].item() == 1.0, (view_name, ch)


def test_full_view_is_index_zero_and_passes_board_through_unchanged() -> None:
    model = ChannelDropoutConsensusNetwork(**_toy_kwargs()).eval()
    assert model.FULL_VIEW_INDEX == 0
    boards = _board_batch([STARTING_FEN])
    views = model._channel_dropped_views(boards)
    assert torch.allclose(views[:, 0], boards)


def test_full_view_only_zeros_variance_and_max_pairwise_features() -> None:
    model = ChannelDropoutConsensusNetwork(**_toy_kwargs(), ablation="full_view_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["variance_latent"], torch.zeros_like(out["variance_latent"]))
    assert torch.allclose(out["max_pairwise"], torch.zeros_like(out["max_pairwise"]))
    # mean is tied to the full latent in full_view_only.
    assert torch.allclose(out["mean_latent"], out["full_view_latent"])


def test_mean_only_zeros_disagreement_features_but_keeps_mean() -> None:
    model = ChannelDropoutConsensusNetwork(**_toy_kwargs(), ablation="mean_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["variance_latent"], torch.zeros_like(out["variance_latent"]))
    assert torch.allclose(out["max_pairwise"], torch.zeros_like(out["max_pairwise"]))
    # mean_only should still average across all views (so mean != full in general).
    assert out["mean_latent"].shape == out["full_view_latent"].shape


def test_random_channel_masks_differs_from_semantic_masks() -> None:
    semantic = ChannelDropoutConsensusNetwork(**_toy_kwargs()).eval()
    randomized = ChannelDropoutConsensusNetwork(**_toy_kwargs(), ablation="random_channel_masks").eval()
    # The full view (index 0) must remain all-ones for both.
    assert torch.allclose(semantic.view_masks[0], torch.ones(18))
    assert torch.allclose(randomized.view_masks[0], torch.ones(18))
    # At least one non-full view must differ to prove the randomization fired.
    assert not torch.allclose(semantic.view_masks[1:], randomized.view_masks[1:])
    # Drop counts must match per view so the comparison is matched-capacity.
    for view_idx, view_name in enumerate(DETERMINISTIC_VIEW_NAMES):
        if view_name == "full":
            continue
        semantic_drops = (semantic.view_masks[view_idx] == 0).sum().item()
        randomized_drops = (randomized.view_masks[view_idx] == 0).sum().item()
        assert semantic_drops == randomized_drops, view_name


def test_train_dropout_only_collapses_to_full_view_at_eval() -> None:
    model = ChannelDropoutConsensusNetwork(**_toy_kwargs(), ablation="train_dropout_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    # In eval mode Dropout2d is a no-op, so every "view" latent equals the full latent.
    expected = out["full_view_latent"].unsqueeze(1).expand(-1, model.num_views, model.channels)
    assert torch.allclose(out["view_pooled"], expected)
    assert torch.allclose(out["mean_latent"], out["full_view_latent"])
    assert torch.allclose(out["variance_latent"], torch.zeros_like(out["variance_latent"]))


def test_all_ablations_run_without_crash_and_emit_their_code() -> None:
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    for ablation in ChannelDropoutConsensusNetwork.ABLATIONS:
        torch.manual_seed(0)
        model = ChannelDropoutConsensusNetwork(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation
        assert out["channel_dropout_ablation"][0].item() == float(
            ChannelDropoutConsensusNetwork.ABLATIONS.index(ablation)
        ), ablation
        assert out["channel_dropout_view_count"][0].item() == float(model.num_views), ablation


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        ChannelDropoutConsensusNetwork(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        ChannelDropoutConsensusNetwork(input_channels=12, num_classes=1)


def test_rejects_multiclass_head_contract() -> None:
    with pytest.raises(ValueError):
        ChannelDropoutConsensusNetwork(input_channels=18, num_classes=2)


def test_shared_encoder_has_expected_output_width() -> None:
    encoder = _SharedBoardEncoder(input_channels=18, channels=8, depth=2, dropout=0.0, use_batchnorm=False)
    out = encoder(torch.zeros(3, 18, 8, 8))
    assert out.shape == (3, 8, 8, 8)
    assert encoder.output_channels == 8


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "i118"
    assert data["slug"] == "channel_dropout_consensus_network"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "i118"
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
    assert isinstance(model, ChannelDropoutConsensusNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.encoder, _SharedBoardEncoder)

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

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i118"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues, conformance_rows[0].issues
