"""Focused tests for the bespoke i066 Bispectral Phase-Coupling Board Network."""
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
from chess_nn_playground.models.trunk.bispectral_phase_coupling import (
    BispectralPhaseCoupling,
    BispectralPhaseCouplingBoardNetwork,
    BispectralPhaseHead,
    BoardFFTFeatureLayer,
    SpectralChannelMixer,
    build_bispectral_phase_coupling_board_network_from_config,
)


REGISTRY_KEY = "bispectral_phase_coupling_board_network"
IDEA_DIR = Path("ideas/registry/i066_bispectral_phase_coupling_board_network")
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_PUZZLE_FEN = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"


def _toy_kwargs(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "input_channels": 18,
        "num_classes": 1,
        "mixed_channels": 8,
        "bispectrum_terms": 16,
        "head_hidden": 32,
        "dropout": 0.0,
        "power_terms": 8,
        "cross_channel_pairs": 4,
        "cross_frequency_terms": 4,
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
    assert isinstance(model, BispectralPhaseCouplingBoardNetwork)
    assert not isinstance(model, ResearchPacketProbe)


def test_builder_accepts_channels_and_hidden_dim_aliases() -> None:
    model = build_bispectral_phase_coupling_board_network_from_config(
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 8,
            "hidden_dim": 32,
            "bispectrum_terms": 12,
            "dropout": 0.0,
        }
    )
    assert model.mixed_channels == 8
    assert model.head.classifier[1].out_features == 32


def test_forward_shape_and_required_keys() -> None:
    model = BispectralPhaseCouplingBoardNetwork(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["logits"].shape == (2,)
    assert torch.isfinite(out["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "bispectral_phase_norm",
        "bispectral_magnitude_mean",
        "power_spectrum_energy",
        "cross_phase_norm",
        "spectral_feature_norm",
        "mixed_field_energy",
        "material_balance",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "bispectral_ablation",
        "bispectral_term_count",
    }
    assert expected_keys.issubset(out)
    for key in expected_keys - {"logits", "prob"}:
        assert out[key].shape[0] == 2, key


def test_single_logit_prob_is_in_unit_interval() -> None:
    model = BispectralPhaseCouplingBoardNetwork(**_toy_kwargs()).eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert out["prob"].shape == (2,)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()


def test_backward_gradients_flow_through_mixer_and_head() -> None:
    model = BispectralPhaseCouplingBoardNetwork(**_toy_kwargs())
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    out = model(boards)
    out["logits"].pow(2).mean().backward()
    mixer_grad = model.mixer.mixer.weight.grad
    assert mixer_grad is not None and torch.isfinite(mixer_grad).all()
    head_first = model.head.classifier[1].weight.grad
    assert head_first is not None and torch.isfinite(head_first).all()


def test_fft_and_frequency_tables_are_buffers_not_parameters() -> None:
    model = BispectralPhaseCouplingBoardNetwork(**_toy_kwargs())
    parameter_ids = {id(p) for p in model.parameters()}
    for name in ("k_rank", "k_file", "l_rank", "l_file", "kl_rank", "kl_file"):
        buf = getattr(model.coupling, name)
        assert id(buf) not in parameter_ids, name
    # The FFT layer carries no parameters at all.
    assert sum(1 for _ in model.fft.parameters()) == 0


def test_magnitude_only_zeros_bispectral_phase_norm() -> None:
    model = BispectralPhaseCouplingBoardNetwork(**_toy_kwargs(), ablation="magnitude_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["bispectral_phase_norm"], torch.zeros_like(out["bispectral_phase_norm"]))
    assert out["bispectral_magnitude_mean"].abs().sum() > 0


def test_power_only_zeros_phase_and_magnitude_features() -> None:
    model = BispectralPhaseCouplingBoardNetwork(**_toy_kwargs(), ablation="power_only").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out = model(boards)
    assert torch.allclose(out["bispectral_phase_norm"], torch.zeros_like(out["bispectral_phase_norm"]))
    assert torch.allclose(out["bispectral_magnitude_mean"], torch.zeros_like(out["bispectral_magnitude_mean"]))
    assert out["power_spectrum_energy"].abs().sum() > 0


def test_phase_batch_shuffle_changes_features_relative_to_full() -> None:
    cfg = _toy_kwargs()
    torch.manual_seed(0)
    full = BispectralPhaseCouplingBoardNetwork(**cfg).eval()
    torch.manual_seed(0)
    shuffled = BispectralPhaseCouplingBoardNetwork(**cfg, ablation="phase_batch_shuffle").eval()
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    with torch.no_grad():
        out_full = full(boards)
        out_shuf = shuffled(boards)
    assert not torch.allclose(out_full["logits"], out_shuf["logits"])


def test_no_coordinate_planes_disables_mixer_coords() -> None:
    model = BispectralPhaseCouplingBoardNetwork(**_toy_kwargs(), ablation="no_coordinate_planes").eval()
    assert model.use_coordinate_planes is False
    assert model.mixer.use_coordinate_planes is False
    # Mixer accepts only the 18 board planes when coord planes are off.
    assert model.mixer.mixer.in_channels == 18


def test_all_ablations_run_without_crash() -> None:
    boards = _board_batch([STARTING_FEN, MATE_PUZZLE_FEN])
    for ablation in BispectralPhaseCouplingBoardNetwork.ABLATIONS:
        torch.manual_seed(0)
        model = BispectralPhaseCouplingBoardNetwork(**_toy_kwargs(), ablation=ablation).eval()
        with torch.no_grad():
            out = model(boards)
        assert torch.isfinite(out["logits"]).all(), ablation
        assert out["bispectral_ablation"][0].item() == float(
            BispectralPhaseCouplingBoardNetwork.ABLATIONS.index(ablation)
        )


def test_rejects_unknown_ablation() -> None:
    with pytest.raises(ValueError):
        BispectralPhaseCouplingBoardNetwork(**_toy_kwargs(), ablation="not_real")


def test_rejects_non_simple_18_input() -> None:
    with pytest.raises(ValueError):
        BispectralPhaseCouplingBoardNetwork(input_channels=12, num_classes=1)


def test_idea_yaml_metadata() -> None:
    data = yaml.safe_load((IDEA_DIR / "idea.yaml").read_text(encoding="utf-8"))
    assert data["idea_id"] == "i066"
    assert data["slug"] == "bispectral_phase_coupling_board_network"
    assert data["implementation_kind"] == "bespoke_model"
    assert data["implementation_status"] == "implemented"
    assert data["status"] == "implemented"


def test_config_yaml_keys() -> None:
    cfg = yaml.safe_load((IDEA_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["idea_id"] == "i066"
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
    assert isinstance(model, BispectralPhaseCouplingBoardNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.mixer, SpectralChannelMixer)
    assert isinstance(model.fft, BoardFFTFeatureLayer)
    assert isinstance(model.coupling, BispectralPhaseCoupling)
    assert isinstance(model.head, BispectralPhaseHead)

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

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i066"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues, conformance_rows[0].issues
