from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
import yaml

from chess_nn_playground.training.config_validation import validate_training_config


CANONICAL_SPLIT_DIR = "data/splits/crtk_sample_3class_unique_crtk_tags"


IDEA_CONTRACTS: dict[str, dict[str, Any]] = {
    "i001": {
        "folder": "ideas/i001_chess_operator_basis_classifier",
        "model": "chess_operator_basis_classifier",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"operator_gate_entropy"},
    },
    "i002": {
        "folder": "ideas/i002_response_minimax_classifier",
        "model": "response_minimax_classifier",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"global_minimax", "reply_entropy", "top_action_gap"},
    },
    "i003": {
        "folder": "ideas/i003_factor_agreement_classifier",
        "model": "factor_agreement_classifier",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"factor_disagreement", "factor_uncertainty", "global_evidence"},
    },
    "i004": {
        "folder": "ideas/i004_puzzle_obligation_flow_network",
        "model": "puzzle_obligation_flow_network",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"allocation_entropy", "flow_residual_max", "flow_residual_mean"},
    },
    "i005": {
        "folder": "ideas/i005_null_move_contrast_puzzle_network",
        "model": "null_move_contrast_puzzle_network",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"current_evidence", "null_evidence", "tempo_contrast_delta"},
    },
    "i006": {
        "folder": "ideas/i006_proof_core_set_verifier",
        "model": "proof_core_set_verifier",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"deletion_gap", "global_residual", "proof_logit", "selection_entropy"},
    },
    "i007": {
        "folder": "ideas/i007_neural_proof_number_search",
        "model": "neural_proof_number_search",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"beam_entropy", "proof_disproof_gap", "root_disproof_cost", "root_proof_cost"},
    },
    "i008": {
        "folder": "ideas/i008_boundary_edit_lagrangian_network",
        "model": "boundary_edit_lagrangian_network",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"E_minus", "E_plus", "base_logit", "edit_gap"},
    },
    "i009": {
        "folder": "ideas/i009_tactical_equilibrium_network",
        "model": "tactical_equilibrium_network",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"attacker_entropy", "defender_entropy", "equilibrium_value", "exploitability"},
    },
    "i010": {
        "folder": "ideas/i010_rule_consistent_latent_dynamics",
        "model": "rule_consistent_latent_dynamics",
        "loss": "bce_with_logits",
        "encoding": "simple_18",
        "diagnostics": {"legal_entropy", "max_transition_norm", "transition_variance"},
    },
    "i011": {
        "folder": "ideas/i011_vetoselect_positive_claim_abstention",
        "model": "vetoselect_positive_claim_abstention",
        "loss": "veto_select",
        "encoding": "lc0_bt4_112",
        "diagnostics": {
            "prob_accepted_puzzle",
            "prob_nonpuzzle",
            "prob_rejected_evidence",
            "selective_puzzle_logit",
            "selector_logit",
        },
    },
    "i012": {
        "folder": "ideas/i012_dykstra_lcp",
        "model": "dykstra_lcp",
        "loss": "dykstra_lcp",
        "encoding": "lc0_bt4_112",
        "diagnostics": {"final_residual", "projection_distance", "trace_residual", "motif_entropy"},
    },
    "i013": {
        "folder": "ideas/i013_sparse_relation_pursuit_asymmetry",
        "model": "sparse_relation_pursuit_asymmetry",
        "loss": "srpa",
        "encoding": "lc0_bt4_112",
        "diagnostics": {
            "aux_logit",
            "branch_separation",
            "dictionary_coherence",
            "residual_asymmetry",
            "tac_final_residual",
        },
    },
    "i014": {
        "folder": "ideas/i014_contamination_dro_huber_tail_rejection",
        "model": "contamination_dro_huber_tail_rejection",
        "loss": "contamination_dro_huber",
        "encoding": "simple_18",
        "diagnostics": {"logit_margin_residual", "material_delta", "material_total"},
    },
    "i015": {
        "folder": "ideas/i015_material_locked_tactical_dro",
        "model": "material_locked_tactical_dro",
        "loss": "material_locked_dro",
        "encoding": "simple_18",
        "diagnostics": {"adversarial_logits", "clean_logits", "mask_budget_used", "tactical_mask_mean"},
    },
    "i016": {
        "folder": "ideas/i016_soft_sorting_order_residual_ranker",
        "model": "soft_sorting_order_residual_ranker",
        "loss": "soft_sort_order",
        "encoding": "simple_18",
        "diagnostics": {"score_scale"},
    },
    "i017": {
        "folder": "ideas/i017_conditional_surprisal_gate",
        "model": "conditional_surprisal_gate",
        "loss": "conditional_surprisal_gate",
        "encoding": "simple_18",
        "diagnostics": {"conditional_surprisal", "gate_mean", "posterior_logits", "prior_logits"},
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_idea_model(folder: Path):
    spec = importlib.util.spec_from_file_location(f"{folder.name}_contract_model", folder / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_static_training_config_valid(path: Path) -> None:
    messages = validate_training_config(_load_yaml(path), path, require_device_available=False)
    errors = [message for message in messages if message.startswith("ERROR:")]
    assert not errors, "\n".join(messages)


def test_all_idea_configs_match_their_research_contracts():
    assert len(IDEA_CONTRACTS) == 17
    for idea_id, contract in IDEA_CONTRACTS.items():
        folder = Path(contract["folder"])
        config_path = folder / "config.yaml"
        _assert_static_training_config_valid(config_path)
        config = _load_yaml(config_path)
        model_cfg = config["model"]
        training_cfg = config["training"]
        data_cfg = config["data"]

        assert config["idea_id"] == idea_id
        assert config["mode"] == "puzzle_binary"
        assert config["device"] == "nvidia"
        assert data_cfg["encoding"] == contract["encoding"]
        assert str(data_cfg["train_path"]).startswith(CANONICAL_SPLIT_DIR)
        assert str(data_cfg["val_path"]).startswith(CANONICAL_SPLIT_DIR)
        assert str(data_cfg["test_path"]).startswith(CANONICAL_SPLIT_DIR)
        assert model_cfg["name"] == contract["model"]
        assert model_cfg["num_classes"] == 1
        assert training_cfg["loss"] == contract["loss"]
        assert training_cfg["class_weighting"] == "balanced"
        assert training_cfg["reliability_tier"] == "paper_grade"
        assert int(training_cfg["epochs"]) >= 20
        assert int(training_cfg["min_epochs"]) >= 10
        assert int(training_cfg["min_active_epochs"]) >= 10


def test_all_idea_models_emit_the_expected_diagnostics():
    for contract in IDEA_CONTRACTS.values():
        folder = Path(contract["folder"])
        config = _load_yaml(folder / "config.yaml")
        module = _load_idea_model(folder)
        model = module.build_model_from_config(config).eval()
        input_channels = int(config["model"]["input_channels"])
        x = torch.zeros(2, input_channels, 8, 8)
        if input_channels > 12:
            x[:, 12] = 1.0

        with torch.no_grad():
            output = model(x)

        assert isinstance(output, dict), f"{folder} should return diagnostics with logits"
        assert output["logits"].shape == (2,), folder
        assert torch.isfinite(output["logits"]).all(), folder
        missing = set(contract["diagnostics"]) - set(output)
        assert not missing, f"{folder} missing diagnostics: {sorted(missing)}"
        for key in contract["diagnostics"]:
            value = output[key]
            assert isinstance(value, torch.Tensor), f"{folder} diagnostic {key} is not a tensor"
            if value.ndim > 0:
                assert value.shape[0] == 2, f"{folder} diagnostic {key} is not batched"
            assert torch.isfinite(value).all(), f"{folder} diagnostic {key} has non-finite values"


def test_all_benchmark_configs_are_static_valid_and_canonical():
    benchmark_paths = sorted(Path("configs/benchmarks").rglob("bench_*.yaml"))
    assert benchmark_paths
    for path in benchmark_paths:
        _assert_static_training_config_valid(path)
        config = _load_yaml(path)
        assert config["device"] == "nvidia"
        assert config["training"]["reliability_tier"] == "paper_grade"
        assert str(config["data"]["train_path"]).startswith(CANONICAL_SPLIT_DIR)
        assert str(config["data"]["val_path"]).startswith(CANONICAL_SPLIT_DIR)
        assert str(config["data"]["test_path"]).startswith(CANONICAL_SPLIT_DIR)
        if "signal" in config["run"]["name"] or path.name in {
            "bench_lc0_bt4_classifier.yaml",
            "bench_mlp_simple18.yaml",
            "bench_nnue_simple18.yaml",
            "bench_srpa_lc0bt4.yaml",
        }:
            assert config["mode"] == "puzzle_binary"
            assert config["model"]["num_classes"] == 1
        if "fine3" in config["run"]["name"]:
            assert config["mode"] == "fine_3class"
            assert config["model"]["num_classes"] == 3


def test_benchmark_suites_reference_valid_configs_with_artifact_validation_enabled():
    suite_expectations = {
        "configs/suites/network_signal_benchmark_suite.yaml": ("puzzle_binary", 1),
        "configs/suites/network_signal_fine3_benchmark_suite.yaml": ("fine_3class", 3),
        "configs/suites/experiment_suite.yaml": ("coarse_binary", 2),
    }
    for suite_path, (expected_mode, expected_classes) in suite_expectations.items():
        suite = _load_yaml(Path(suite_path))
        assert suite["validate_artifacts"] is True
        assert suite["rebuild_leaderboard"] is True
        assert suite["seeds"] == [42, 43, 44]
        for config_ref in suite["configs"]:
            config_path = Path(config_ref)
            assert config_path.exists()
            _assert_static_training_config_valid(config_path)
            config = _load_yaml(config_path)
            assert config["mode"] == expected_mode
            assert config["model"]["num_classes"] == expected_classes
