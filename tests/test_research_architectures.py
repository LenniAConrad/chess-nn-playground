from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import yaml

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from chess_nn_playground.data.board_features import fen_to_tensor
from chess_nn_playground.ideas.architecture_conformance import audit_architecture_conformance
from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.ideas.implementation_kind import analyze_model_wiring
from chess_nn_playground.ideas.implementation_kind import detect_idea_implementation_kind
from chess_nn_playground.models.registry import MODEL_BUILDERS
from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.hall_defect_obligation_matroid import HallZetaDefectLayer
from chess_nn_playground.models.hall_defect_obligation_matroid import ObligationBatch
from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


@lru_cache(maxsize=1)
def _audit_architecture_conformance_rows():
    return tuple(audit_architecture_conformance())


REGISTERED_RESEARCH_ARCHITECTURES = {
    "chess_operator_basis_classifier": {
        "input_channels": 18,
        "hidden_dim": 16,
        "blocks": 1,
        "relation_operators": ["identity", "knight", "king"],
    },
    "response_minimax_classifier": {
        "input_channels": 18,
        "board_channels": 16,
        "token_dim": 16,
        "max_actions": 6,
        "max_replies_per_action": 4,
    },
    "factor_agreement_classifier": {"input_channels": 18, "branch_dim": 16},
    "puzzle_obligation_flow_network": {
        "input_channels": 18,
        "trunk_channels": 16,
        "token_dim": 16,
        "max_obligations": 6,
        "max_resources": 8,
        "solver_steps": 2,
    },
    "null_move_contrast_puzzle_network": {
        "input_channels": 18,
        "encoder_channels": 16,
        "latent_dim": 16,
        "pair_mixer_layers": 1,
    },
    "proof_core_set_verifier": {
        "input_channels": 18,
        "token_dim": 16,
        "relation_dim": 8,
        "max_tokens": 32,
        "selected_k": 4,
    },
    "neural_proof_number_search": {
        "input_channels": 18,
        "board_channels": 16,
        "latent_dim": 16,
        "or_beam": 4,
        "and_beam": 3,
        "transition_layers": 1,
    },
    "boundary_edit_lagrangian_network": {
        "input_channels": 18,
        "encoder_channels": 16,
        "latent_dim": 16,
        "max_edits": 8,
        "solver_steps": 2,
        "edit_feature_dim": 8,
    },
    "tactical_equilibrium_network": {
        "input_channels": 18,
        "trunk_channels": 16,
        "token_dim": 16,
        "relation_dim": 8,
        "max_attackers": 4,
        "max_defenders": 5,
        "solver_steps": 2,
    },
    "rule_consistent_latent_dynamics": {
        "input_channels": 18,
        "encoder_channels": 16,
        "latent_dim": 16,
        "move_feature_dim": 8,
        "max_moves": 6,
        "max_invalid": 4,
        "transition_layers": 1,
    },
    "vetoselect_positive_claim_abstention": {
        "input_channels": 112,
        "channels": 8,
        "num_blocks": 1,
        "value_channels": 4,
        "value_hidden": 16,
        "se_channels": 4,
        "use_batchnorm": False,
    },
    "dykstra_lcp": {
        "input_channels": 112,
        "channels": 8,
        "num_blocks": 1,
        "value_channels": 4,
        "value_hidden": 16,
        "se_channels": 4,
        "role_count": 4,
        "relation_channels": 2,
        "motif_count": 2,
        "slack_count": 2,
        "solver_cycles": 1,
        "use_batchnorm": False,
    },
    "sparse_relation_pursuit_asymmetry": {
        "input_channels": 112,
        "square_dim": 8,
        "stem_depth": 1,
        "relation_dim": 8,
        "geom_dim": 4,
        "path_dim": 4,
        "num_atom_groups": 2,
        "atoms_per_group": 2,
        "pursuit_steps": 1,
        "classifier_hidden": 8,
        "max_ray_distance": 2,
        "edge_chunk_size": 256,
        "use_batchnorm": False,
    },
    "contamination_dro_huber_tail_rejection": {
        "input_channels": 18,
        "channels": 16,
        "depth": 1,
        "hidden_dim": 16,
        "use_batchnorm": False,
    },
    "material_locked_tactical_dro": {
        "input_channels": 18,
        "channels": 16,
        "depth": 1,
        "hidden_dim": 16,
        "rho": 0.05,
        "use_batchnorm": False,
    },
    "soft_sorting_order_residual_ranker": {
        "input_channels": 18,
        "channels": 16,
        "depth": 1,
        "hidden_dim": 16,
        "use_batchnorm": False,
    },
    "conditional_surprisal_gate": {
        "input_channels": 18,
        "channels": 16,
        "depth": 1,
        "hidden_dim": 16,
        "gate_dim": 8,
        "hard_gate": False,
        "use_batchnorm": False,
    },
    "king_escape_percolation_network": {
        "input_channels": 18,
        "cost_hidden_dim": 8,
        "escape_taus": [0.5],
        "escape_steps": 3,
        "dp_snapshots": [1, 2, 3],
        "stem_width": 8,
        "fusion_width": 12,
        "classifier_hidden_dim": 16,
        "dropout": 0.0,
        "use_batchnorm": False,
    },
    "soft_king_cage_path_bottleneck_network": {
        "input_channels": 18,
        "trunk_width": 8,
        "trunk_blocks": 1,
        "barrier_hidden_channels": 8,
        "dp_radii": [2, 3],
        "dp_temperatures": [0.5],
        "dp_steps": 3,
        "dp_big_m": 20.0,
        "use_distance_fields": True,
        "hidden_dim": 16,
        "dropout": 0.0,
        "use_batchnorm": False,
    },
    "hall_defect_obligation_matroid_network": {
        "input_channels": 18,
        "channels": 8,
        "hidden_dim": 16,
        "depth": 1,
        "d_max_defenders": 4,
        "o_max_obligations": 16,
        "lambdas": [1.0, 2.0],
        "token_dim": 8,
        "dropout": 0.0,
        "use_batchnorm": False,
    },
    "threat_topology_betti_bottleneck_network": {
        "input_channels": 18,
        "channels": 8,
        "hidden_dim": 16,
        "topology_hidden_dim": 16,
        "topology_embedding_dim": 8,
        "rank_ks": [1, 2, 4, 8],
        "dropout": 0.0,
        "use_batchnorm": False,
    },
    "blocker_pin_lattice_network": {
        "input_channels": 18,
        "channels": 16,
        "hidden_dim": 24,
        "depth": 1,
        "ray_dim": 16,
        "lattice_states": 4,
        "layers": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
    },
    "safe_reply_certificate_verifier": {
        "input_channels": 18,
        "channels": 16,
        "hidden_dim": 24,
        "depth": 1,
        "certificate_dim": 24,
        "max_certificates": 24,
        "dropout": 0.0,
        "use_batchnorm": False,
    },
    "latent_reply_entropy_network": {
        "input_channels": 18,
        "channels": 16,
        "hidden_dim": 24,
        "depth": 1,
        "reply_dim": 24,
        "temperature": 0.7,
        "max_replies": 24,
        "dropout": 0.0,
        "use_batchnorm": False,
    },
    "tactical_threat_sheaf_network": {
        "input_channels": 18,
        "d_model": 16,
        "hidden_dim": 24,
        "num_sheaf_layers": 1,
        "max_edges": 512,
        "relation_type_count": 96,
        "restriction_rank": 2,
        "gate_hidden": 24,
        "classifier_hidden": 24,
        "dropout": 0.0,
        "edge_dropout": 0.0,
        "use_edge_gates": True,
        "use_contest_pool": True,
        "use_square_embeddings": True,
    },
    "attack_hodge_sheaf_tension_network": {
        "input_channels": 18,
        "d_model": 16,
        "hidden_dim": 24,
        "n_layers": 1,
        "transport_rank": 2,
        "max_edges": 512,
        "max_faces": 512,
        "edge_type_count": 96,
        "dropout": 0.0,
        "edge_dropout": 0.0,
        "use_xray_edges": True,
        "use_face_hodge": True,
        "use_energy_pool": True,
        "classifier_hidden": 24,
    },
    "directed_attack_sheaf_tension_network": {
        "input_channels": 18,
        "d_model": 16,
        "hidden_dim": 24,
        "num_layers": 1,
        "restriction_rank": 2,
        "max_edges": 512,
        "edge_type_count": 128,
        "dropout": 0.0,
        "edge_dropout": 0.0,
        "use_xray_edges": True,
        "classifier_hidden": 24,
    },
    "one_ply_counterfactual_move_landscape_network": {
        "input_channels": 18,
        "root_channels": 8,
        "root_embedding_dim": 16,
        "move_dim": 16,
        "hidden_dim": 24,
        "depth": 1,
        "max_moves": 64,
        "landscape_temperature": 0.5,
        "use_count_scalar": False,
        "include_path_summary": False,
        "include_castling_candidates": False,
        "adapter_strict": True,
        "classifier_hidden": 24,
        "dropout": 0.0,
    },
    "hypercolumn_square_readout_cnn": {
        "input_channels": 18,
        "trunk_width": 16,
        "trunk_depth": 2,
        "hyper_width": 8,
        "evidence_width": 8,
        "hidden_dim": 24,
        "topk_squares": 2,
        "dropout": 0.0,
        "use_batchnorm": False,
        "ablation": "none",
    },
    "multiplicative_conjunction_convnet": {
        "input_channels": 18,
        "width": 16,
        "depth": 2,
        "branch_width": 8,
        "hidden_dim": 24,
        "dropout": 0.0,
        "use_batchnorm": False,
        "use_coordinate_planes": True,
        "ablation": "none",
    },
    "empty_square_opportunity_network": {
        "input_channels": 18,
        "trunk_width": 16,
        "branch_width": 12,
        "opportunity_channels": 4,
        "hidden_dim": 24,
        "fusion_width": 24,
        "depth": 2,
        "topk_squares": 2,
        "dropout": 0.0,
        "use_batchnorm": False,
        "use_coordinate_planes": True,
        "ablation": "none",
    },
    "global_scratchpad_boardnet": {
        "input_channels": 18,
        "width": 16,
        "memory_slots": 3,
        "memory_dim": 12,
        "scratchpad_steps": 2,
        "hidden_dim": 24,
        "dropout": 0.0,
        "use_batchnorm": False,
        "use_coordinate_planes": True,
        "ablation": "none",
    },
    "independence_residual_interaction_network": {
        "input_channels": 18,
        "channels": 16,
        "hidden_dim": 24,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "expected_mix": 0.5,
    },
    "residual_calibration_error_field": {
        "input_channels": 18,
        "channels": 16,
        "hidden_dim": 24,
        "depth": 1,
        "dropout": 0.0,
        "use_batchnorm": False,
        "error_channels": 4,
        "temperature_floor": 0.25,
        "correction_scale": 1.0,
    },
    "set_query_attention_bottleneck": {
        "input_channels": 18,
        "token_dim": 16,
        "query_count": 6,
        "head_count": 4,
        "hidden_dim": 24,
        "head_hidden": 24,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "include_attention_diagnostics": True,
        "ablation": "none",
    },
    "variational_board_action_network": {
        "input_channels": 18,
        "field_channels": 4,
        "context_width": 8,
        "channels": 8,
        "hidden_dim": 16,
        "depth": 1,
        "residual_map_width": 8,
        "dropout": 0.0,
        "use_batchnorm": False,
        "boundary_mode": "reflect",
        "include_residual_map_cnn": True,
        "include_board_cnn_summary": True,
        "use_force_head_approximation": True,
        "ablation": "none",
    },
    "tensor_core_square_pair_field_network": {
        "input_channels": 18,
        "model_dim": 32,
        "heads": 4,
        "head_dim": 8,
        "layers": 1,
        "pair_rank": 4,
        "hidden_dim": 32,
        "classifier_hidden": 32,
        "ffn_multiplier": 2,
        "dropout": 0.0,
        "norm": "rmsnorm",
        "activation": "gelu",
        "ablation": "none",
    },
    "tiny_chess_micronet": {
        "input_channels": 18,
        "width": 8,
        "squeeze_rank": 4,
        "blocks": 1,
        "mix_rank": 3,
        "head_hidden": 8,
        "dropout": 0.0,
        "line_bases": ["constant", "center_heavy", "side_relative_forward"],
        "king_zone": False,
        "quantization_target": "int8",
        "ablation": "none",
    },
}


def _load_idea_model(folder: Path):
    spec = importlib.util.spec_from_file_location(f"{folder.name}_research_model", folder / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remaining_research_architectures_are_registered():
    assert set(REGISTERED_RESEARCH_ARCHITECTURES).issubset(set(available_models()))


def test_promoted_research_packets_have_named_profiled_builders():
    builder_names = {MODEL_BUILDERS[name].__name__ for name in RESEARCH_PACKET_MODEL_NAMES}
    assert len(builder_names) == len(RESEARCH_PACKET_MODEL_NAMES)

    branches = set()
    signatures = set()
    for name in RESEARCH_PACKET_MODEL_NAMES:
        model = build_model(
            name,
            {
                "input_channels": 18,
                "num_classes": 1,
                "channels": 8,
                "hidden_dim": 16,
                "depth": 1,
                "dropout": 0.0,
                "use_batchnorm": False,
            },
        )
        assert isinstance(model, ResearchPacketProbe)
        assert model.cfg.packet_profile == name
        branches.add(model.mechanism_key)
        signatures.add(tuple(model.profile_signature.tolist()))

    assert len(branches) >= 10
    assert len(signatures) == len(RESEARCH_PACKET_MODEL_NAMES)


def test_remaining_research_architectures_forward_shape_and_diagnostics():
    for name, config in REGISTERED_RESEARCH_ARCHITECTURES.items():
        input_channels = int(config["input_channels"])
        x = torch.zeros(2, input_channels, 8, 8)
        if input_channels > 12:
            x[:, 12] = 1.0
        model = build_model(
            name,
            {
                "num_classes": 1,
                "dropout": 0.0,
                **config,
            },
        )

        output = model(x)

        assert isinstance(output, dict), name
        assert output["logits"].shape == (2,), name
        assert torch.isfinite(output["logits"]).all(), name
        diagnostics = [value for key, value in output.items() if key != "logits" and isinstance(value, torch.Tensor)]
        assert diagnostics, name
        assert all(torch.isfinite(value).all() for value in diagnostics), name


def test_remaining_idea_configs_are_trainable():
    for folder in sorted(Path("ideas").glob("i[0-9][0-9][0-9]_*")):
        idea = yaml.safe_load((folder / "idea.yaml").read_text(encoding="utf-8"))
        if idea.get("implementation_status") not in {"implemented", "tested"}:
            continue
        report = validate_idea_for_training(folder)
        assert report["valid"], report


def test_i070_relational_query_algebra_is_bespoke_and_conformant():
    folder = Path("ideas/i070_relational_query_algebra_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "relational_query_algebra_network"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 1, 4] = 1.0
    x[:, 3, 3, 4] = 1.0
    x[:, 5, 0, 4] = 1.0
    x[:, 6, 6, 4] = 1.0
    x[:, 10, 4, 4] = 1.0
    x[:, 11, 7, 4] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "relation_mixture_entropy",
        "query_support_entropy",
        "piece_square_join_strength",
        "piece_piece_join_strength",
        "semijoin_strength",
        "cnn_energy",
        "material_balance",
        "piece_count",
    }.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    pieces = model.piece_extractor(x)
    squares = model.square_builder(x)
    assert pieces.features.shape[:2] == (2, 32)
    assert pieces.mask.sum(dim=1).le(32).all()
    assert squares.features.shape[:2] == (2, 64)
    assert model.relation_bank.shape == (config["model"]["relation_count"], 64, 64)
    assert model.between_line_mask[4, 60, 12] == 1
    assert model.between_line_mask[4, 60, 20] == 1

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i070"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i071_variational_board_action_is_bespoke_and_conformant():
    folder = Path("ideas/i071_variational_board_action_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "variational_board_action_network"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 2, 5, 2] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 10, 2, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0

    constant = torch.ones(2, model.field_channels, 8, 8)
    dx, dy = model.finite_difference(constant)
    assert dx.shape == constant.shape
    assert dy.shape == constant.shape
    assert torch.allclose(dx, torch.zeros_like(dx))
    assert torch.allclose(dy, torch.zeros_like(dy))
    assert model.finite_difference.adjoint(dx, dy).shape == constant.shape

    with torch.no_grad():
        fields = model.field_encoder(x)
        gx, gy, force, density = model.lagrangian_heads(fields.fields, fields.context)
        terms = model.residual_layer(fields.fields, gx, gy, force, density)
        output = model(x)

    assert terms.residual.shape == fields.fields.shape
    assert torch.isfinite(terms.residual).all()
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "action_value",
        "potential_energy",
        "gradient_energy",
        "residual_l1",
        "residual_l2",
        "residual_max",
        "king_zone_residual",
        "occupied_square_residual",
        "empty_square_residual",
        "boundary_flux",
        "residual_map_energy",
        "dx_energy",
        "dy_energy",
    }.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    for ablation in [
        "cnn_only_matched",
        "action_only",
        "no_gradient_terms",
        "random_difference_operators",
        "residual_norm_only",
        "force_head_only",
        "harmonic_control",
    ]:
        ablated_cfg = {
            "input_channels": 18,
            "num_classes": 1,
            "field_channels": 4,
            "context_width": 8,
            "channels": 8,
            "hidden_dim": 16,
            "depth": 1,
            "residual_map_width": 8,
            "dropout": 0.0,
            "use_batchnorm": False,
            "ablation": ablation,
        }
        ablated = build_model(model_name, ablated_cfg).eval()
        with torch.no_grad():
            ablated_output = ablated(x)
        assert ablated_output["logits"].shape == (2,), ablation
        assert torch.isfinite(ablated_output["logits"]).all(), ablation

    trainable = build_model(
        model_name,
        {
            "input_channels": 18,
            "num_classes": 1,
            "field_channels": 4,
            "context_width": 8,
            "channels": 8,
            "hidden_dim": 16,
            "depth": 1,
            "residual_map_width": 8,
            "dropout": 0.0,
            "use_batchnorm": False,
        },
    )
    grad_output = trainable(x)["logits"].sum()
    grad_output.backward()
    assert trainable.field_encoder.field_head.weight.grad is not None
    assert torch.isfinite(trainable.field_encoder.field_head.weight.grad).all()
    assert trainable.lagrangian_heads.gx_head.weight.grad is not None
    assert torch.isfinite(trainable.lagrangian_heads.gx_head.weight.grad).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i071"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i072_tensor_core_square_pair_field_is_bespoke_and_conformant():
    folder = Path("ideas/i072_tensor_core_square_pair_field_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "tensor_core_square_pair_field_network"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 3, 5, 2] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 9, 2, 5] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0

    relation_bank = model.relation_bank
    shuffled_bank = model.shuffled_relation_bank
    assert relation_bank.shape == (18, 64, 64)
    assert torch.allclose(relation_bank, module.TensorCoreSquarePairFieldNetwork().relation_bank)
    assert torch.allclose(relation_bank.mean(dim=(1, 2)), shuffled_bank.mean(dim=(1, 2)))
    assert torch.allclose(relation_bank.diagonal(dim1=1, dim2=2).sum(dim=1), shuffled_bank.diagonal(dim1=1, dim2=2).sum(dim=1))
    assert torch.equal(shuffled_bank[1], shuffled_bank[1].T)

    same_rank, same_file, same_diag, same_anti_diag = relation_bank[1], relation_bank[2], relation_bank[3], relation_bank[4]
    knight_offset, king_offset = relation_bank[7], relation_bank[8]
    rank_forward, file_forward = relation_bank[16], relation_bank[17]
    assert same_rank[0, 7] == 1
    assert same_rank[0, 8] == 0
    assert same_file[0, 8] == 1
    assert same_file[0, 1] == 0
    assert same_diag[0, 9] == 1
    assert same_anti_diag[7, 14] == 1
    assert knight_offset[0, 17] == 1
    assert king_offset[0, 9] == 1
    assert rank_forward[0, 8] == 1
    assert rank_forward[8, 0] == 0
    assert file_forward[0, 1] == 1
    assert file_forward[1, 0] == 0

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "same_rank_pair_energy",
        "same_file_pair_energy",
        "diagonal_pair_energy",
        "knight_offset_pair_energy",
        "occupied_to_occupied_pair_energy",
        "occupied_to_empty_pair_energy",
        "king_zone_pair_energy",
        "pair_field_entropy_proxy",
        "per_head_energy_specialization",
        "relation_density_error",
    }.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key
    assert torch.allclose(output["relation_density_error"], torch.zeros_like(output["relation_density_error"]))

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    for ablation in [
        "cnn_only_matched",
        "no_pair_update",
        "no_pair_readout",
        "relation_bank_shuffle",
        "softmax_attention_control",
        "low_head_count",
        "pair_energy_only",
    ]:
        ablated = build_model(
            model_name,
            {
                "input_channels": 18,
                "num_classes": 1,
                "model_dim": 32,
                "heads": 4,
                "head_dim": 8,
                "layers": 1,
                "pair_rank": 4,
                "hidden_dim": 32,
                "classifier_hidden": 32,
                "ffn_multiplier": 2,
                "dropout": 0.0,
                "ablation": ablation,
            },
        ).eval()
        with torch.no_grad():
            ablated_output = ablated(x)
        assert ablated_output["logits"].shape == (2,), ablation
        assert torch.isfinite(ablated_output["logits"]).all(), ablation

    trainable = build_model(
        model_name,
        {
            "input_channels": 18,
            "num_classes": 1,
            "model_dim": 32,
            "heads": 4,
            "head_dim": 8,
            "layers": 1,
            "pair_rank": 4,
            "hidden_dim": 32,
            "classifier_hidden": 32,
            "ffn_multiplier": 2,
            "dropout": 0.0,
        },
    )
    trainable(x)["logits"].sum().backward()
    assert trainable.blocks[0].qkv.weight.grad is not None
    assert torch.isfinite(trainable.blocks[0].qkv.weight.grad).all()
    assert trainable.blocks[0].relation_weight.grad is not None
    assert torch.isfinite(trainable.blocks[0].relation_weight.grad).all()

    if torch.cuda.is_available():
        cuda_model = build_model(
            model_name,
            {
                "input_channels": 18,
                "num_classes": 1,
                "model_dim": 32,
                "heads": 4,
                "head_dim": 8,
                "layers": 1,
                "pair_rank": 4,
                "hidden_dim": 32,
                "classifier_hidden": 32,
                "dropout": 0.0,
            },
        ).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cuda_output = cuda_model(x.cuda())
        assert cuda_output["logits"].shape == (2,)
        assert torch.isfinite(cuda_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i072"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i073_tiny_chess_micronet_is_bespoke_and_conformant():
    folder = Path("ideas/i073_tiny_chess_micronet")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "tiny_chess_micronet"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 1, 6, 3] = 1.0
    x[:, 3, 5, 2] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 9, 2, 5] = 1.0
    x[:, 10, 1, 2] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0

    assert model.sketch.global_dim == 3 * config["model"]["width"]
    assert model.sketch.line_dim == 4 * 6 * config["model"]["width"]
    assert model.sketch.king_dim == 6 * config["model"]["width"]
    assert model.sketch.material_dim == 18
    assert model.blocks[0].line_back.shape == (4, 64, 64)
    assert torch.allclose(model.blocks[0].line_back[0].sum(dim=1), torch.ones(64))
    assert torch.allclose(model.blocks[0].line_back[1].sum(dim=1), torch.ones(64))
    assert model.blocks[0].line_back[0, 0, 7] > 0
    assert model.blocks[0].line_back[0, 0, 8] == 0
    assert model.blocks[0].line_back[1, 0, 8] > 0
    assert model.blocks[0].line_back[2, 0, 9] > 0
    assert model.blocks[0].line_back[3, 7, 14] > 0

    with torch.no_grad():
        h = model._field(x)
        groups = model.sketch(h, x)
        output = model(x)

    assert groups.global_pool.shape == (2, model.sketch.global_dim)
    assert groups.line_sketch.shape == (2, model.sketch.line_dim)
    assert groups.king_zone.shape == (2, model.sketch.king_dim)
    assert groups.material.shape == (2, 18)
    assert torch.equal(groups.malformed_king_count, torch.zeros_like(groups.malformed_king_count))
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["parameter_count"].max() <= 25000
    assert {
        "hidden_field_energy",
        "global_descriptor_energy",
        "line_sketch_energy",
        "king_zone_energy",
        "material_summary_energy",
        "global_pool_norm_fraction",
        "line_sketch_norm_fraction",
        "king_zone_norm_fraction",
        "material_norm_fraction",
        "parameter_count",
        "fp32_size_bytes",
        "simulated_int8_size_bytes",
        "malformed_king_count",
    }.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    malformed = x.clone()
    malformed[:, 5] = 0.0
    with torch.no_grad():
        malformed_output = model(malformed)
    assert (malformed_output["malformed_king_count"] >= 1).all()
    assert torch.isfinite(malformed_output["logits"]).all()

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    for ablation in [
        "counts_only_mlp",
        "ordinary_tiny_cnn_matched",
        "flat_head_same_params",
        "no_line_sketch",
        "random_line_basis",
        "no_king_zone",
        "no_depthwise_local",
    ]:
        ablated = build_model(
            model_name,
            {
                "input_channels": 18,
                "num_classes": 1,
                "width": 8,
                "squeeze_rank": 4,
                "blocks": 1,
                "mix_rank": 3,
                "head_hidden": 8,
                "dropout": 0.0,
                "line_bases": ["constant", "center_heavy", "side_relative_forward"],
                "king_zone": True,
                "ablation": ablation,
            },
        ).eval()
        with torch.no_grad():
            ablated_output = ablated(x)
        assert ablated_output["logits"].shape == (2,), ablation
        assert torch.isfinite(ablated_output["logits"]).all(), ablation

    trainable = build_model(
        model_name,
        {
            "input_channels": 18,
            "num_classes": 1,
            "width": 8,
            "squeeze_rank": 4,
            "blocks": 1,
            "mix_rank": 3,
            "head_hidden": 8,
            "dropout": 0.0,
            "line_bases": ["constant", "center_heavy", "side_relative_forward"],
            "king_zone": True,
        },
    )
    trainable(x)["logits"].sum().backward()
    assert trainable.squeeze.net[0].weight.grad is not None
    assert torch.isfinite(trainable.squeeze.net[0].weight.grad).all()
    assert trainable.blocks[0].line_gamma.grad is not None
    assert torch.isfinite(trainable.blocks[0].line_gamma.grad).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in audit_architecture_conformance() if row.idea_id == "i073"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i051_king_escape_percolation_is_bespoke_and_conformant():
    folder = Path("ideas/i051_king_escape_percolation_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 1, 7, 1] = 1.0
    x[:, 3, 4, 4] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 5] = 1.0
    x[:, 8, 0, 2] = 1.0
    x[:, 10, 0, 3] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0
    x[:, 13] = 1.0

    geometry = model.adapter(x)
    assert geometry.pieces.shape == (2, 2, 6, 8, 8)
    assert geometry.king_masks.shape == (2, 2, 8, 8)
    assert geometry.occupancy.shape == (2, 8, 8)
    assert torch.allclose(geometry.king_masks[:, 0, 7, 4], torch.ones(2))
    assert torch.allclose(geometry.king_masks[:, 1, 0, 4], torch.ones(2))

    attack_counts, attack_type_counts = model.attack_maps(geometry.pieces, geometry.occupancy)
    assert attack_counts.shape == (2, 2, 1, 8, 8)
    assert attack_type_counts.shape == (2, 2, 6, 8, 8)
    assert (attack_type_counts[:, 0, 3, 0, 4] > 0).all()

    cost_fields, attacker_counts, defender_counts = model.cost_field(
        geometry.pieces,
        geometry.side_to_move_white,
        geometry.king_masks,
        geometry.occupancy,
        attack_counts,
        attack_type_counts,
    )
    assert cost_fields.shape == (2, 2, 8, 8)
    assert attacker_counts.shape == defender_counts.shape == (2, 2, 8, 8)
    assert torch.isfinite(cost_fields).all()
    assert (cost_fields > 0).all()

    with torch.no_grad():
        output = model(x)
        aux = model(x, return_aux=True)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "mechanism_energy",
        "topology_pressure",
        "king_ring_pressure",
        "escape_edge_energy",
        "escape_reachable_mass",
        "escape_asymmetry",
        "defense_gap",
        "cost_field_mean",
    }.issubset(output)
    assert output["two_class_logits"].shape == (2, 2)
    assert aux["escape_maps"].shape == (2, model.escape_dp.map_channels, 8, 8)
    assert aux["escape_vec"].shape == (2, model.escape_dp.vector_dim)
    assert aux["attack_counts"].shape == (2, 2, 1, 8, 8)
    assert aux["cost_fields"].shape == (2, 2, 8, 8)
    assert torch.isfinite(aux["escape_maps"]).all()
    assert torch.isfinite(aux["escape_vec"]).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i051"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i052_soft_king_cage_path_is_bespoke_and_conformant():
    folder = Path("ideas/i052_soft_king_cage_path_bottleneck_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 1, 7, 1] = 1.0
    x[:, 3, 4, 4] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 5] = 1.0
    x[:, 8, 0, 2] = 1.0
    x[:, 10, 0, 3] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0
    x[:, 13] = 1.0

    parsed = model.adapter(x)
    assert parsed.pieces.shape == (2, 2, 6, 8, 8)
    assert parsed.king_maps.shape == (2, 2, 8, 8)
    assert torch.allclose(parsed.king_maps[:, 0, 7, 4], torch.ones(2))
    assert torch.allclose(parsed.king_maps[:, 1, 0, 4], torch.ones(2))

    geom = model.rule_geometry(parsed)
    assert geom.attack_counts.shape == (2, 2, 8, 8)
    assert geom.opponent_attack_pressure.shape == (2, 2, 8, 8)
    assert (geom.attack_counts[:, 0, 0, 4] > 0).all()

    barrier, barrier_diag = model.barrier_field(geom)
    assert barrier.shape == (2, 2, 8, 8)
    assert torch.isfinite(barrier).all()
    assert (barrier >= 0).all()
    assert {"attack_weight", "own_occupancy_weight", "opponent_occupancy_weight"}.issubset(barrier_diag)
    assert (barrier_diag["attack_weight"] > 0).all()

    fields, scalars, dp_diag = model.escape_dp(barrier, geom.king_maps)
    assert fields.shape == (2, 2, len(model.escape_dp.dp_radii), len(model.escape_dp.dp_temperatures), 8, 8)
    assert scalars.shape == (2, 2, len(model.escape_dp.dp_radii), len(model.escape_dp.dp_temperatures))
    assert dp_diag["target_shell_mass"].shape == (2, len(model.escape_dp.dp_radii))
    assert torch.isfinite(fields).all()
    assert torch.isfinite(scalars).all()

    real_degree = model.escape_dp.grid_neighbor_mask.sum(dim=1)
    random_degree = model.escape_dp.random_neighbor_mask.sum(dim=1)
    assert torch.equal(real_degree, random_degree)

    with torch.no_grad():
        output = model(x)
        aux = model(x, return_aux=True)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["two_class_logits"].shape == (2, 2)
    assert {
        "cage_energy",
        "side_to_move_cage_gap",
        "topology_pressure",
        "king_ring_pressure",
        "path_entropy_proxy",
        "cage_asymmetry",
        "barrier_mean",
        "attack_barrier_weight",
        "occupancy_barrier_weight",
        "defense_gap",
    }.issubset(output)
    assert aux["distance_fields"].shape == fields.shape
    assert aux["cage_scalars"].shape == scalars.shape
    assert aux["cage_features"].shape[0] == 2
    assert torch.isfinite(aux["cage_features"]).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i052"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i053_hall_defect_obligation_matroid_is_bespoke_and_conformant():
    folder = Path("ideas/i053_hall_defect_obligation_matroid_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 1, 7, 1] = 1.0
    x[:, 3, 4, 4] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 5] = 1.0
    x[:, 8, 0, 2] = 1.0
    x[:, 10, 0, 3] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0

    decoded = model.attack_generator(x)
    assert decoded.pieces.shape == (2, 2, 6, 8, 8)
    assert decoded.piece_slots.shape[:3] == (2, 2, 16)
    assert decoded.controls.shape == (2, 2, 16, 64)
    assert decoded.attack_count.shape == (2, 2, 64)
    assert (decoded.attack_count[:, 0, 4] > 0).all()

    obligations = model.obligation_builder(x)
    assert obligations.obligation_masks.shape[:3] == (2, 2, 6)
    assert obligations.neighborhood_bitmasks.shape == obligations.obligation_masks.shape
    assert obligations.defender_masks.shape[:3] == (2, 2, 6)
    assert torch.isfinite(obligations.obligation_weights).all()
    assert (obligations.obligation_masks.sum(dim=(1, 2, 3)) > 0).all()

    hall_tokens, hall_diag = model.hall_zeta(obligations)
    assert hall_tokens.shape[:3] == (2, 2, 6)
    assert hall_tokens.shape[-1] == model.hall_zeta.token_dim
    assert hall_diag["hist_count"].shape[:3] == (2, 2, 6)
    assert torch.isfinite(hall_tokens).all()
    assert torch.isfinite(hall_diag["weighted_defects"]).all()

    manual_layer = HallZetaDefectLayer(d_max_defenders=2, lambdas=(1.0,))
    manual = ObligationBatch(
        obligation_masks=torch.tensor([[[[1.0, 1.0, 0.0]]]]),
        obligation_weights=torch.tensor([[[[1.0, 1.0, 0.0]]]]),
        neighborhood_bitmasks=torch.tensor([[[[1, 1, 0]]]], dtype=torch.long),
        defender_masks=torch.tensor([[[[1.0, 1.0]]]]),
        num_defenders_total=torch.tensor([[[2.0]]]),
        num_defenders_discarded=torch.tensor([[[0.0]]]),
        edge_counts=torch.tensor([[[2.0]]]),
        degree_sums=torch.tensor([[[2.0]]]),
        max_degrees=torch.tensor([[[1.0]]]),
        zero_degree_counts=torch.tensor([[[0.0]]]),
    )
    _, manual_diag = manual_layer(manual)
    assert torch.allclose(manual_diag["cardinal_defect"], torch.ones(1, 1, 1))
    assert torch.allclose(manual_diag["weighted_defects"][..., 0], torch.ones(1, 1, 1))

    with torch.no_grad():
        output = model(x)
        aux = model(x, return_aux=True)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "hall_cardinal_defect",
        "hall_mean_cardinal_defect",
        "hall_weighted_defect",
        "hall_defect_energy",
        "sparse_certificate_energy",
        "overload_role_gap",
        "defense_gap",
        "obligation_count",
        "defender_count",
        "defender_truncation_count",
        "hall_edge_density",
        "zero_defender_obligation_count",
        "board_context_energy",
        "mechanism_energy",
        "proposal_profile_strength",
    }.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key
    assert aux["hall_tokens"].shape == hall_tokens.shape
    assert aux["neighborhood_bitmasks"].shape == obligations.neighborhood_bitmasks.shape
    assert aux["hist_count"].shape == hall_diag["hist_count"].shape
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    for ablation in ["degree_rewire", "count_only", "weight_shuffle", "complete_neighborhood"]:
        ablated = build_model(
            model_name,
            {
                "input_channels": 18,
                "num_classes": 1,
                "channels": 8,
                "hidden_dim": 16,
                "d_max_defenders": 4,
                "o_max_obligations": 16,
                "lambdas": [1.0],
                "token_dim": 8,
                "dropout": 0.0,
                "use_batchnorm": False,
                "edge_ablation_mode": ablation,
            },
        ).eval()
        with torch.no_grad():
            ablated_output = ablated(x)
        assert ablated_output["logits"].shape == (2,), ablation
        assert torch.isfinite(ablated_output["logits"]).all(), ablation

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i053"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i085_hall_defect_zeta_operator_is_bespoke_and_conformant():
    from chess_nn_playground.models.hall_defect_zeta import HallDefectZetaConvLite

    folder = Path("ideas/i085_hall_defect_zeta_operator")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, HallDefectZetaConvLite)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "hall_defect_zeta_operator"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — produces a non-trivial pin/overload pattern on a1 and king ring.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["hdz_tensor"].shape == (2, 40, 8, 8)
    assert output["hdz_only_logits"].shape == (2,)
    for key in (
        "zeta_defect_spectrum",
        "max_hall_defect",
        "mean_hall_defect",
        "hall_defect_energy",
        "effective_defense_density",
        "pinned_defender_density",
        "loose_target_density",
        "loose_target_count",
        "pinned_piece_count",
        "effective_defense_total",
        "mechanism_energy",
        "proposal_profile_strength",
    ):
        assert key in output, key
        assert torch.isfinite(output[key]).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    assert isinstance(registry_model, HallDefectZetaConvLite)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    for algebra_mode in ("hdz", "atom_scramble_hdz", "neural_synth_40"):
        ablated = HallDefectZetaConvLite(
            input_channels=18,
            num_classes=1,
            channels=8,
            hidden_dim=16,
            depth=1,
            algebra_mode=algebra_mode,
            max_subset_order=2,
            max_atoms=8,
            dropout=0.0,
            use_batchnorm=False,
        ).eval()
        with torch.no_grad():
            ablated_output = ablated(x)
        assert ablated_output["logits"].shape == (2,), algebra_mode
        assert torch.isfinite(ablated_output["logits"]).all(), algebra_mode

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i085"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i055_non_backtracking_tactical_walk_is_bespoke_and_conformant():
    folder = Path("ideas/i055_non_backtracking_tactical_walk_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    # white king e1, black king e8, white rook a1, black rook a8, white pawn e2,
    # black knight c6 — produces a real edge graph with attack/protection chains.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 9, 0, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 7, 2, 2] = 1.0

    with torch.no_grad():
        output = model(x)
        aux = model(x, return_aux=True)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["two_class_logits"].shape == (2, 2)
    assert {
        "non_backtracking_walk_energy",
        "edge_count",
        "transition_count",
        "edge_overflow_count",
        "transition_overflow_count",
        "enemy_attack_edge_count",
        "friendly_protect_edge_count",
        "enemy_king_zone_edge_count",
        "own_king_zone_edge_count",
        "mechanism_energy",
        "defense_gap",
        "king_ring_pressure",
    }.issubset(output)
    assert (output["edge_count"] > 0).all()
    assert (output["edge_overflow_count"] == 0).all()
    assert aux["edge_features"].shape == (2, model.edge_max, 32)
    assert aux["edge_state"].shape == (2, model.edge_max, model.edge_dim)
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")

    # Non-backtracking transition correctness: turning the no-immediate-return
    # exclusion off must add the immediate reverse transitions the default mode
    # forbids (e.g., WR -> BR -> WR and BR -> WR -> BR are now allowed).
    backtracking_allowed = build_model(
        model_name,
        {
            "input_channels": 18,
            "num_classes": 1,
            "edge_dim": int(config["model"]["edge_dim"]),
            "edge_layers": int(config["model"]["edge_layers"]),
            "edge_max": int(config["model"]["edge_max"]),
            "transition_max": int(config["model"]["transition_max"]),
            "r_basis": int(config["model"]["r_basis"]),
            "board_adapter_channels": int(config["model"]["board_adapter_channels"]),
            "classifier_hidden_dim": int(config["model"]["classifier_hidden_dim"]),
            "dropout": 0.0,
            "use_batchnorm": False,
            "ablation_mode": "backtracking_allowed",
        },
    ).eval()
    with torch.no_grad():
        bt_output = backtracking_allowed(x)
    # backtracking_allowed must strictly add transitions (the immediate reverses).
    assert (bt_output["transition_count"] >= output["transition_count"]).all()
    assert (bt_output["transition_count"] > output["transition_count"]).any()

    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    for ablation in ("backtracking_allowed", "randomized_transitions"):
        ablated = build_model(
            model_name,
            {
                "input_channels": 18,
                "num_classes": 1,
                "edge_dim": 16,
                "edge_layers": 2,
                "edge_max": 64,
                "transition_max": 256,
                "r_basis": 2,
                "board_adapter_channels": 8,
                "classifier_hidden_dim": 16,
                "dropout": 0.0,
                "use_batchnorm": False,
                "ablation_mode": ablation,
            },
        ).eval()
        with torch.no_grad():
            ablated_output = ablated(x)
        assert ablated_output["logits"].shape == (2,), ablation
        assert torch.isfinite(ablated_output["logits"]).all(), ablation

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i055"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i054_threat_topology_betti_is_bespoke_and_conformant():
    folder = Path("ideas/i054_threat_topology_betti_bottleneck_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 3, 4, 4] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 3] = 1.0
    x[:, 8, 2, 2] = 1.0
    x[:, 10, 0, 6] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0

    decoded = model.adapter(x)
    assert decoded.pieces.shape == (2, 2, 6, 8, 8)
    assert decoded.king_maps.shape == (2, 2, 8, 8)
    assert torch.allclose(decoded.king_maps[:, 0, 7, 4], torch.ones(2))
    assert torch.allclose(decoded.king_maps[:, 1, 0, 4], torch.ones(2))

    pressure = model.rule_pressure(decoded)
    assert pressure.attack_pressure.shape == (2, 2, 8, 8)
    assert pressure.pressure_fields.shape == (2, 4, 8, 8)
    assert (pressure.attack_pressure[:, 0, 0, 4] > 0).all()

    masks = model.betti_encoder.topk_masks(pressure.pressure_fields)
    topology_features = model.betti_encoder.features_from_masks(masks, pressure.pressure_fields)
    assert masks.shape == (2, 4, len(model.betti_encoder.rank_ks), 8, 8)
    assert topology_features.shape == (2, 4, len(model.betti_encoder.rank_ks), 4)
    assert torch.isfinite(topology_features).all()

    block_mask = torch.zeros(1, 1, 1, 8, 8, dtype=torch.bool)
    block_mask[..., 0:2, 0:2] = True
    corner_mask = torch.zeros_like(block_mask)
    corner_mask[..., 0, 0] = True
    corner_mask[..., 0, 7] = True
    corner_mask[..., 7, 0] = True
    corner_mask[..., 7, 7] = True
    assert model.betti_encoder.beta0(block_mask).item() == 1.0
    assert model.betti_encoder.beta0(corner_mask).item() == 4.0

    ring_mask = torch.zeros(1, 1, 1, 8, 8, dtype=torch.bool)
    ring_mask[..., 0:3, 0:3] = True
    ring_mask[..., 1, 1] = False
    ring_stats = model.betti_encoder.features_from_masks(ring_mask, torch.ones(1, 1, 8, 8))
    assert ring_stats[..., 1].item() == 1.0

    field = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8)
    assert torch.equal(model.betti_encoder.topk_masks(field), model.betti_encoder.topk_masks(2.0 * field + 5.0))

    with torch.no_grad():
        output = model(x)
        aux = model(x, return_aux=True)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["two_class_logits"].shape == (2, 2)
    assert {
        "topology_pressure",
        "betti0_mean",
        "betti1_mean",
        "boundary_edge_mean",
        "topk_pressure_mean",
        "pressure_surplus_energy",
        "king_ring_pressure",
        "mechanism_energy",
        "defense_gap",
    }.issubset(output)
    assert aux["pressure_fields"].shape == (2, 4, 8, 8)
    assert aux["topology_features"].shape == topology_features.shape
    assert aux["topk_masks"].shape == masks.shape
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    for ablation in ("rank_shuffle", "histogram_only", "no_topology_fusion"):
        ablated = build_model(
            model_name,
            {
                "input_channels": 18,
                "num_classes": 1,
                "channels": 8,
                "hidden_dim": 16,
                "topology_hidden_dim": 16,
                "topology_embedding_dim": 8,
                "rank_ks": [1, 2, 4],
                "dropout": 0.0,
                "use_batchnorm": False,
                "topology_ablation": ablation,
            },
        ).eval()
        with torch.no_grad():
            ablated_output = ablated(x)
        assert ablated_output["logits"].shape == (2,), ablation
        assert torch.isfinite(ablated_output["logits"]).all(), ablation

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i054"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i022_tactical_threat_sheaf_is_bespoke_and_conformant():
    folder = Path("ideas/i022_tactical_threat_sheaf_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "sheaf_tension",
        "attack_energy",
        "defense_energy",
        "pin_energy",
        "contest_pressure",
        "overload_pressure",
        "gate_mean",
        "edge_density",
    }.issubset(output)
    assert (output["edge_density"] > 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model_cfg["encoding"] = config["data"]["encoding"]
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i022"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i018_oriented_tactical_sheaf_laplacian_is_bespoke_and_conformant():
    folder = Path("ideas/i018_oriented_tactical_sheaf_laplacian")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "sheaf_tension",
        "transport_imbalance",
        "symmetry_residual",
        "topology_pressure",
        "ray_language_energy",
        "king_ring_pressure",
        "reply_pressure",
        "defense_gap",
        "triad_defect_energy",
        "pin_pressure",
    }.issubset(output)
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model_cfg["encoding"] = config["data"]["encoding"]
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i018"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i019_tactical_sheaf_curvature_is_bespoke_and_conformant():
    folder = Path("ideas/i019_tactical_sheaf_curvature_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "sheaf_frustration",
        "curvature_mean",
        "curvature_max",
        "gate_mean",
        "gate_entropy",
        "ray_energy",
        "jump_energy",
        "pawn_candidate_energy",
        "relation_gate_pressure",
        "node_stalk_std",
    }.issubset(output)
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i019"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i021_tactical_sheaf_tension_is_bespoke_and_conformant():
    folder = Path("ideas/i021_tactical_sheaf_tension_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "sheaf_tension",
        "weighted_sheaf_tension",
        "max_edge_tension",
        "top3_edge_tension",
        "edge_density",
        "control_energy",
        "attack_energy",
        "defense_energy",
        "xray_energy",
        "king_ring_energy",
        "side_piece_count",
        "opponent_piece_count",
    }.issubset(output)
    assert (output["edge_density"] > 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model_cfg["encoding"] = config["data"]["encoding"]
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i021"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i020_attack_defense_sheaf_is_bespoke_and_conformant():
    folder = Path("ideas/i020_attack_defense_sheaf_energy_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "sheaf_tension",
        "mechanism_energy",
        "ray_visibility_mean",
        "gate_mean",
        "edge_energy_mean",
        "ray_energy",
        "knight_energy",
        "king_energy",
        "pawn_energy",
        "convergence_tension",
        "defense_gap",
        "top_edge_tension",
        "occupancy_proxy_mean",
    }.issubset(output)
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i020"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i023_attack_hodge_sheaf_is_bespoke_and_conformant():
    folder = Path("ideas/i023_attack_hodge_sheaf_tension_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "sheaf_tension",
        "hodge_edge_tension",
        "node_edge_energy",
        "face_curl_energy",
        "attack_energy",
        "defense_energy",
        "xray_energy",
        "fork_fan_energy",
        "overload_sink_energy",
        "ray_pin_energy",
        "edge_density",
        "face_density",
    }.issubset(output)
    assert (output["edge_density"] > 0).all()
    assert (output["face_density"] > 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model_cfg["encoding"] = config["data"]["encoding"]
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i023"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i024_directed_attack_sheaf_is_bespoke_and_conformant():
    folder = Path("ideas/i024_directed_attack_sheaf_tension_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "sheaf_tension",
        "directed_asymmetry",
        "outgoing_tension",
        "incoming_tension",
        "one_way_tension",
        "reciprocal_tension",
        "xray_tension",
        "king_zone_tension",
        "attack_energy",
        "defense_energy",
        "gate_mean",
        "edge_density",
    }.issubset(output)
    assert (output["edge_density"] > 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model_cfg["encoding"] = config["data"]["encoding"]
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i024"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i025_one_ply_counterfactual_move_landscape_is_bespoke_and_conformant():
    folder = Path("ideas/i025_one_ply_counterfactual_move_landscape_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "move_landscape_free_energy",
        "move_landscape_entropy",
        "move_energy_mean",
        "move_energy_max",
        "move_energy_top2_gap",
        "move_attention_peak",
        "pseudo_legal_move_count",
        "capture_move_fraction",
        "promotion_move_fraction",
    }.issubset(output)
    assert (output["pseudo_legal_move_count"] > 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model_cfg["encoding"] = config["data"]["encoding"]
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i025"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i026_counterfactual_move_delta_spectrum_is_bespoke_and_conformant():
    folder = Path("ideas/i026_counterfactual_move_delta_spectrum_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 3, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "spectrum_trace",
        "spectrum_leading_fraction",
        "spectrum_participation_ratio",
        "spectrum_entropy",
        "spectrum_frobenius_norm",
        "spectrum_top_eigenvalue",
        "spectrum_response_mean_norm",
        "spectrum_response_max_norm",
        "spectrum_response_var_sum",
        "pseudo_legal_move_count",
        "capture_move_fraction",
        "promotion_move_fraction",
    }
    assert expected_keys.issubset(output)
    assert (output["pseudo_legal_move_count"] > 0).all()
    assert (output["spectrum_leading_fraction"] >= 0.0).all()
    assert (output["spectrum_leading_fraction"] <= 1.0 + 1.0e-4).all()
    assert (output["spectrum_trace"] >= 0.0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model_cfg["encoding"] = config["data"]["encoding"]
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i026"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i027_rule_only_counterfactual_move_delta_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i027_rule_only_counterfactual_move_delta_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 5, 0] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "move_cone_kappa",
        "move_cone_score_max",
        "move_cone_score_logmeanexp",
        "move_cone_sparse_active_count",
        "move_cone_alpha_max",
        "move_cone_alpha_entropy",
        "move_cone_b_sparse_norm",
        "move_cone_b_mean_norm",
        "move_cone_b_second_sum",
        "pseudo_legal_move_count",
        "capture_move_fraction",
        "promotion_move_fraction",
    }
    assert expected_keys.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert value.shape == (2,) or value.shape == (2, output["logits"].shape[-1] if value.dim() > 1 else 0), key
            assert torch.isfinite(value).all(), key
    assert (output["pseudo_legal_move_count"] > 0).all()
    assert (output["move_cone_alpha_max"] >= 0.0).all()
    assert (output["move_cone_alpha_max"] <= 1.0 + 1.0e-4).all()
    assert (output["move_cone_sparse_active_count"] >= 1.0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model_cfg["encoding"] = config["data"]["encoding"]
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i027"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i127_square_color_parity_mixer_is_bespoke_and_conformant():
    folder = Path("ideas/i127_square_color_parity_mixer")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 1, 5, 5] = 1.0
    x[:, 2, 4, 2] = 1.0
    x[:, 4, 3, 4] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 7, 2, 2] = 1.0
    x[:, 8, 3, 3] = 1.0
    x[:, 11, 0, 7] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "within_gate_mean",
        "cross_gate_mean",
        "bishop_within_gate",
        "knight_cross_gate",
        "pawn_cross_gate",
        "queen_within_gate",
        "within_block_energy",
        "cross_block_energy",
        "cross_within_ratio",
        "dark_token_energy",
        "light_token_energy",
        "dark_light_energy_gap",
        "dark_block_norm",
        "light_block_norm",
        "cross_block_norm",
    }.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert value.shape == (2,), key
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i127"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i128_occupancy_run_length_segment_encoder_is_bespoke_and_conformant():
    folder = Path("ideas/i128_occupancy_run_length_segment_encoder")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 3] = 1.0
    x[:, 2, 4, 1] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 1] = 1.0
    x[:, 9, 0, 4] = 1.0
    x[:, 10, 1, 6] = 1.0
    x[:, 11, 0, 7] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "segment_branch_energy",
        "rank_segment_energy",
        "file_segment_energy",
        "diagonal_segment_energy",
        "anti_diagonal_segment_energy",
        "rank_line_contribution",
        "file_line_contribution",
        "diagonal_line_contribution",
        "anti_diagonal_line_contribution",
        "empty_run_mean",
        "occupied_run_mean",
        "open_segment_fraction",
        "king_zone_segment_fraction",
        "king_slider_gap_mean",
        "segment_count_mean",
        "endpoint_type_entropy",
    }.issubset(output)
    assert (output["segment_count_mean"] > 0).all()
    assert (output["king_slider_gap_mean"] > 0).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert value.shape == (2,), key
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i128"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i129_king_shelter_microkernel_network_is_bespoke_and_conformant():
    folder = Path("ideas/i129_king_shelter_microkernel_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[0, 12] = 1.0
    x[0, 0, 6, 3:6] = 1.0
    x[0, 3, 7, 0] = 1.0
    x[0, 4, 5, 4] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 8, 3, 2] = 1.0
    x[0, 9, 0, 4] = 1.0
    x[0, 10, 1, 6] = 1.0
    x[0, 11, 0, 6] = 1.0

    x[1, 6, 1, 2:5] = 1.0
    x[1, 9, 0, 7] = 1.0
    x[1, 10, 2, 3] = 1.0
    x[1, 11, 0, 3] = 1.0
    x[1, 2, 4, 5] = 1.0
    x[1, 3, 7, 3] = 1.0
    x[1, 4, 6, 1] = 1.0
    x[1, 5, 7, 5] = 1.0

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "king_crop_branch_logit",
        "own_microkernel_energy",
        "opponent_microkernel_energy",
        "king_zone_residual",
        "front_shield_score",
        "side_escape_score",
        "diagonal_entry_pressure",
        "rank_backdoor_pressure",
        "near_slider_pressure",
        "local_blocker_density",
        "king_ring_escape",
        "king_zone_density",
        "local_pressure",
        "shelter_escape_gap",
        "opponent_front_shield_score",
        "opponent_local_pressure",
        "shelter_residual",
        "crop5_activation_energy",
        "crop7_activation_energy",
    }.issubset(output)
    assert (output["own_microkernel_energy"] > 0).all()
    assert (output["crop5_activation_energy"] > 0).all()
    assert (output["crop7_activation_energy"] > 0).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert value.shape == (2,), key
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i129"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i160_hypercolumn_square_readout_is_bespoke_and_conformant():
    folder = Path("ideas/i160_hypercolumn_square_readout_cnn")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 3, 4, 4] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 8, 0, 4] = 1.0
    x[:, 11, 0, 7] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["square_logits"].shape == (2, 2, 8, 8)
    assert output["square_puzzle_evidence"].shape == (2, 8, 8)
    assert output["evidence_map"].shape[0] == 2
    assert output["evidence_map"].shape[-2:] == (8, 8)
    assert {
        "hypercolumn_energy",
        "evidence_energy",
        "square_logit_energy",
        "top_square_evidence",
        "top_square_index",
        "layer_projection_energy",
        "early_projection_energy",
        "late_projection_energy",
        "late_over_early_projection",
        "aggregate_feature_energy",
    }.issubset(output)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i160"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i190_blocker_pin_lattice_is_bespoke_and_conformant():
    folder = Path("ideas/i190_blocker_pin_lattice_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    fen = "4k3/8/8/8/4p3/8/4R3/4K3 w - - 0 1"
    x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).repeat(2, 1, 1, 1)
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "pin_strength",
        "discovered_attack_potential",
        "blocked_tactic_residual",
        "lattice_energy",
        "pin_lattice_entropy",
        "ray_count",
        "ordered_blocker_mass",
        "state_current_strength",
        "state_remove_first_strength",
        "state_remove_second_strength",
        "state_swap_side_strength",
    }.issubset(output)
    assert (output["pin_strength"] > 0).all()
    assert (output["discovered_attack_potential"] > 0).all()
    assert (output["blocked_tactic_residual"] > 0).all()
    assert (output["ray_count"] > 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i190"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i191_safe_reply_certificate_verifier_is_bespoke_and_conformant():
    folder = Path("ideas/i191_safe_reply_certificate_verifier")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    fen = "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).repeat(2, 1, 1, 1)
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "positive_puzzle_logit",
        "best_disproof",
        "certificate_validity",
        "certificate_strength",
        "certificate_score",
        "certificate_valid",
        "certificate_kind",
        "safe_reply_certificate_count",
        "validity_mean",
        "strength_mean",
        "move_away_king_certificate",
        "capture_attacker_certificate",
        "block_line_certificate",
        "defend_target_certificate",
        "counter_threat_certificate",
        "trade_down_certificate",
        "certificate_kind_count",
        "certificate_kind_max",
    }
    assert expected_keys.issubset(output)
    max_certificates = int(config["model"]["max_certificates"])
    for key in {"certificate_validity", "certificate_strength", "certificate_score", "certificate_valid", "certificate_kind"}:
        assert output[key].shape == (2, max_certificates), key
    assert output["certificate_kind_count"].shape == (2, 6)
    assert output["certificate_kind_max"].shape == (2, 6)
    assert (output["safe_reply_certificate_count"] > 0).all()
    assert (output["best_disproof"] > 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i191"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i192_latent_reply_entropy_network_is_bespoke_and_conformant():
    folder = Path("ideas/i192_latent_reply_entropy_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    fen = "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).repeat(2, 1, 1, 1)
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "reply_entropy",
        "reply_entropy_normalized",
        "reply_top1",
        "reply_top2_gap",
        "effective_reply_count",
        "valid_reply_count",
        "safe_reply_mass",
        "reply_score",
        "reply_probability",
        "reply_valid",
        "reply_kind",
        "reply_source_square",
        "reply_target_square",
        "reply_kind_count",
        "reply_kind_max",
        "reply_kind_probability",
        "reply_kind_score",
        "king_escape_reply_mass",
        "capture_attacker_reply_mass",
        "block_line_reply_mass",
        "defend_target_reply_mass",
        "counter_threat_reply_mass",
        "quiet_resource_reply_mass",
    }
    assert expected_keys.issubset(output)
    max_replies = int(config["model"]["max_replies"])
    for key in {"reply_score", "reply_probability", "reply_valid", "reply_kind", "reply_source_square", "reply_target_square"}:
        assert output[key].shape == (2, max_replies), key
    assert output["reply_kind_count"].shape == (2, 6)
    assert output["reply_kind_max"].shape == (2, 6)
    assert output["reply_kind_probability"].shape == (2, 6)
    assert torch.allclose(output["reply_probability"].sum(dim=1), torch.ones(2), atol=1.0e-5)
    assert (output["valid_reply_count"] > 0).all()
    assert (output["reply_entropy"] >= 0).all()
    assert (output["effective_reply_count"] >= 1.0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i192"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i145_piece_plane_gated_cnn_is_bespoke_and_conformant():
    folder = Path("ideas/i145_piece_plane_gated_cnn")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 1, 7, 6] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 10, 0, 3] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0
    x[:, 13] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "gate_white_mean",
        "gate_black_mean",
        "gate_state_mean",
        "gate_entropy",
        "white_piece_count",
        "black_piece_count",
        "state_signal",
        "semantic_grouping_known",
        "trunk_feature_energy",
        "pooled_feature_std",
    }
    assert expected_keys.issubset(output)
    for key in expected_keys:
        assert output[key].shape == (2,)
        assert torch.isfinite(output[key]).all()
    for key in {"gate_white_mean", "gate_black_mean", "gate_state_mean"}:
        assert (output[key] >= 0).all()
        assert (output[key] <= 1).all()
    assert torch.equal(output["semantic_grouping_known"], torch.ones(2))
    assert (output["trunk_feature_energy"] >= 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    registry_model = build_model(
        "piece_plane_gated_cnn",
        {
            "input_channels": 18,
            "num_classes": 1,
            "group_width": 8,
            "trunk_width": 24,
            "trunk_depth": 1,
            "stem_depth": 1,
            "gate_hidden": 16,
            "hidden_dim": 32,
            "dropout": 0.0,
            "use_batchnorm": False,
        },
    ).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert isinstance(registry_output, dict)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    no_gates = build_model(
        "piece_plane_gated_cnn",
        {
            "input_channels": 18,
            "num_classes": 1,
            "group_width": 8,
            "trunk_width": 24,
            "trunk_depth": 1,
            "stem_depth": 1,
            "gate_hidden": 16,
            "hidden_dim": 32,
            "dropout": 0.0,
            "use_batchnorm": False,
            "ablation": "no_gates",
        },
    ).eval()
    with torch.no_grad():
        no_gates_output = no_gates(x)
    assert no_gates_output["logits"].shape == (2,)
    assert torch.equal(no_gates_output["gate_white_mean"], torch.ones(2))
    assert torch.equal(no_gates_output["gate_black_mean"], torch.ones(2))
    assert torch.equal(no_gates_output["gate_state_mean"], torch.ones(2))

    random_groups = build_model(
        "piece_plane_gated_cnn",
        {
            "input_channels": 18,
            "num_classes": 1,
            "group_width": 8,
            "trunk_width": 24,
            "trunk_depth": 1,
            "stem_depth": 1,
            "gate_hidden": 16,
            "hidden_dim": 32,
            "dropout": 0.0,
            "use_batchnorm": False,
            "ablation": "random_channel_groups",
        },
    ).eval()
    with torch.no_grad():
        random_output = random_groups(x)
    assert random_output["logits"].shape == (2,)
    assert torch.equal(random_output["semantic_grouping_known"], torch.zeros(2))

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i145"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i146_patch_mixer_boardnet_is_bespoke_and_conformant():
    folder = Path("ideas/i146_patch_mixer_boardnet")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 1, 7, 6] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 10, 0, 3] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "patch_token_energy",
        "patch_token_std",
        "token_mixing_energy",
        "channel_mixing_energy",
        "pooled_patch_contrast",
        "patch_occupancy_mean",
        "active_patch_fraction",
        "patch_count",
        "patch_size",
    }
    assert expected_keys.issubset(output)
    for key in expected_keys:
        assert output[key].shape == (2,)
        assert torch.isfinite(output[key]).all()
    assert torch.equal(output["patch_count"], torch.full((2,), 16.0))
    assert torch.equal(output["patch_size"], torch.full((2,), 2.0))
    assert (output["patch_token_energy"] >= 0).all()
    assert (output["token_mixing_energy"] >= 0).all()
    assert (output["channel_mixing_energy"] >= 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    registry_model = build_model(
        "patch_mixer_boardnet",
        {
            "input_channels": 18,
            "num_classes": 1,
            "patch_size": 2,
            "token_count": 16,
            "embed_dim": 24,
            "depth": 1,
            "token_mlp_dim": 16,
            "channel_mlp_dim": 32,
            "hidden_dim": 32,
            "dropout": 0.0,
        },
    ).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert isinstance(registry_output, dict)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    no_token_mixing = build_model(
        "patch_mixer_boardnet",
        {
            "input_channels": 18,
            "num_classes": 1,
            "patch_size": 2,
            "token_count": 16,
            "embed_dim": 24,
            "depth": 1,
            "token_mlp_dim": 16,
            "channel_mlp_dim": 32,
            "hidden_dim": 32,
            "dropout": 0.0,
            "ablation": "no_token_mixing",
        },
    ).eval()
    with torch.no_grad():
        no_token_output = no_token_mixing(x)
    assert no_token_output["logits"].shape == (2,)
    assert torch.equal(no_token_output["token_mixing_energy"], torch.zeros(2))
    assert (no_token_output["channel_mixing_energy"] >= 0).all()

    patch1 = build_model(
        "patch_mixer_boardnet",
        {
            "input_channels": 18,
            "num_classes": 1,
            "patch_size": 2,
            "token_count": 16,
            "embed_dim": 24,
            "depth": 1,
            "token_mlp_dim": 16,
            "channel_mlp_dim": 32,
            "hidden_dim": 32,
            "dropout": 0.0,
            "ablation": "patch1_square_mixer",
        },
    ).eval()
    with torch.no_grad():
        patch1_output = patch1(x)
    assert patch1_output["logits"].shape == (2,)
    assert torch.equal(patch1_output["patch_count"], torch.full((2,), 64.0))
    assert torch.equal(patch1_output["patch_size"], torch.ones(2))

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i146"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i147_specialist_head_cnn_is_bespoke_and_conformant():
    folder = Path("ideas/i147_specialist_head_cnn")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 1, 7, 6] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 10, 0, 3] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "learned_fusion_logit",
        "uniform_average_logit",
        "global_head_logit",
        "center_head_logit",
        "edge_head_logit",
        "king_head_logit",
        "material_head_logit",
        "active_head_count",
        "specialist_logit_std",
        "global_logit_share",
        "center_logit_share",
        "edge_logit_share",
        "king_logit_share",
        "material_logit_share",
        "trunk_feature_energy",
        "global_feature_energy",
        "center_region_energy",
        "edge_region_energy",
        "king_zone_decoded",
        "own_king_zone_mass",
        "opponent_king_zone_mass",
        "material_balance",
        "material_phase",
        "piece_count_total",
    }
    assert expected_keys.issubset(output)
    for key in expected_keys:
        assert output[key].shape == (2,), key
        assert torch.isfinite(output[key]).all(), key
    assert torch.equal(output["king_zone_decoded"], torch.ones(2))
    assert torch.equal(output["active_head_count"], torch.full((2,), 5.0))
    assert (output["own_king_zone_mass"] > 0).all()
    assert (output["opponent_king_zone_mass"] > 0).all()
    assert (output["trunk_feature_energy"] >= 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    registry_model = build_model(
        "specialist_head_cnn",
        {
            "input_channels": 18,
            "num_classes": 1,
            "trunk_width": 16,
            "trunk_depth": 1,
            "head_hidden": 8,
            "fusion_hidden": 16,
            "dropout": 0.0,
            "use_batchnorm": False,
        },
    ).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert isinstance(registry_output, dict)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    uniform_average = build_model(
        "specialist_head_cnn",
        {
            "input_channels": 18,
            "num_classes": 1,
            "trunk_width": 16,
            "trunk_depth": 1,
            "head_hidden": 8,
            "fusion_hidden": 16,
            "dropout": 0.0,
            "use_batchnorm": False,
            "ablation": "uniform_logit_average",
        },
    ).eval()
    with torch.no_grad():
        uniform_output = uniform_average(x)
    assert torch.allclose(uniform_output["logits"], uniform_output["uniform_average_logit"])

    no_king = build_model(
        "specialist_head_cnn",
        {
            "input_channels": 18,
            "num_classes": 1,
            "trunk_width": 16,
            "trunk_depth": 1,
            "head_hidden": 8,
            "fusion_hidden": 16,
            "dropout": 0.0,
            "use_batchnorm": False,
            "ablation": "no_king_head",
        },
    ).eval()
    with torch.no_grad():
        no_king_output = no_king(x)
    assert no_king_output["logits"].shape == (2,)
    assert torch.equal(no_king_output["king_head_logit"], torch.zeros(2))
    assert torch.equal(no_king_output["active_head_count"], torch.full((2,), 4.0))

    single_global = build_model(
        "specialist_head_cnn",
        {
            "input_channels": 18,
            "num_classes": 1,
            "trunk_width": 16,
            "trunk_depth": 1,
            "head_hidden": 8,
            "fusion_hidden": 16,
            "dropout": 0.0,
            "use_batchnorm": False,
            "ablation": "single_global_head",
        },
    ).eval()
    with torch.no_grad():
        single_output = single_global(x)
    assert torch.equal(single_output["active_head_count"], torch.ones(2))
    assert torch.allclose(single_output["logits"], single_output["global_head_logit"])

    empty = torch.zeros_like(x)
    empty[:, 12] = 1.0
    with torch.no_grad():
        empty_output = registry_model(empty)
    assert torch.equal(empty_output["king_zone_decoded"], torch.zeros(2))
    assert torch.equal(empty_output["king_head_logit"], torch.zeros(2))

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i147"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i100_independence_residual_interaction_network_is_bespoke_and_conformant():
    folder = Path("ideas/i100_independence_residual_interaction_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    fen = "4k3/8/8/3q4/4R3/8/4Q3/4K3 w - - 0 1"
    x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).repeat(2, 1, 1, 1)
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "residual_l1",
        "residual_l2",
        "positive_residual_mass",
        "negative_residual_mass",
        "expected_mass_ratio",
        "piece_entropy",
        "square_entropy",
        "rank_file_coupling",
        "interaction_energy",
        "signed_channel_coupling",
        "material_balance",
        "occupancy_count",
    }.issubset(output)
    for key in (
        "residual_l1",
        "residual_l2",
        "positive_residual_mass",
        "negative_residual_mass",
        "expected_mass_ratio",
        "rank_file_coupling",
        "interaction_energy",
        "signed_channel_coupling",
    ):
        assert output[key].shape == (2,)
        assert torch.isfinite(output[key]).all(), key
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    tiny = build_model(
        "independence_residual_interaction_network",
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 16,
            "hidden_dim": 24,
            "depth": 1,
            "dropout": 0.0,
            "use_batchnorm": False,
            "expected_mix": 0.5,
        },
    ).eval()
    with torch.no_grad():
        tiny_output = tiny(x)
    assert tiny_output["logits"].shape == (2,)
    assert torch.isfinite(tiny_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i100"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i101_residual_calibration_error_field_is_bespoke_and_conformant():
    folder = Path("ideas/i101_residual_calibration_error_field")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    fen = "4k3/8/8/3q4/4R3/8/4Q3/4K3 w - - 0 1"
    x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).repeat(2, 1, 1, 1)
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "raw_logit",
        "calibration_temperature",
        "calibration_correction",
        "correction_norm",
        "correction_regularizer",
        "temperature_log",
        "raw_probability",
        "calibrated_probability",
        "confidence_delta",
        "calibration_strength",
        "error_field",
        "error_field_energy",
        "error_field_peak",
        "error_field_l1",
        "error_field_entropy",
        "error_field_center_mass",
        "error_field_edge_mass",
        "error_field_signed_mean",
    }.issubset(output)
    for key in (
        "raw_logit",
        "calibration_temperature",
        "calibration_correction",
        "correction_norm",
        "correction_regularizer",
        "temperature_log",
        "raw_probability",
        "calibrated_probability",
        "confidence_delta",
        "calibration_strength",
        "error_field_energy",
        "error_field_peak",
        "error_field_l1",
        "error_field_entropy",
        "error_field_center_mass",
        "error_field_edge_mass",
        "error_field_signed_mean",
    ):
        assert output[key].shape == (2,)
        assert torch.isfinite(output[key]).all(), key
    assert output["error_field"].shape[0] == 2
    assert output["error_field"].shape[2:] == (8, 8)
    assert torch.isfinite(output["error_field"]).all()
    assert (output["calibration_temperature"] > 0).all()
    assert (output["correction_norm"] >= 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    tiny = build_model(
        "residual_calibration_error_field",
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 16,
            "hidden_dim": 24,
            "depth": 1,
            "dropout": 0.0,
            "use_batchnorm": False,
            "error_channels": 4,
            "temperature_floor": 0.25,
            "correction_scale": 1.0,
        },
    ).eval()
    with torch.no_grad():
        tiny_output = tiny(x)
    assert tiny_output["logits"].shape == (2,)
    assert tiny_output["error_field"].shape == (2, 4, 8, 8)
    assert torch.isfinite(tiny_output["logits"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i101"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i102_set_query_attention_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i102_set_query_attention_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    fen = "4k3/8/8/3q4/4R3/8/4Q3/4K3 w - - 0 1"
    x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).repeat(2, 1, 1, 1)
    with torch.no_grad():
        output = model(x)

    query_count = int(config["model"].get("query_count", 24))
    token_dim = int(config["model"].get("token_dim", config["model"].get("channels", 64)))
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["attention"].shape == (2, query_count, 64)
    assert output["attended_values"].shape == (2, query_count, token_dim)
    assert output["query_diagnostics"].shape == (2, query_count, 11)
    assert torch.allclose(output["attention"].sum(dim=-1), torch.ones(2, query_count), atol=1.0e-5)
    assert {
        "attention_entropy_mean",
        "attention_entropy_std",
        "attention_max_mean",
        "attention_margin_mean",
        "occupied_attention_mass",
        "empty_attention_mass",
        "own_piece_attention_mass",
        "opponent_piece_attention_mass",
        "attended_coord_rank_mean",
        "attended_coord_file_mean",
        "attended_coord_rank_var",
        "attended_coord_file_var",
        "query_diversity",
        "attended_value_norm",
        "token_feature_energy",
        "side_to_move_white",
    }.issubset(output)
    for key in (
        "attention_entropy_mean",
        "attention_entropy_std",
        "attention_max_mean",
        "attention_margin_mean",
        "occupied_attention_mass",
        "empty_attention_mass",
        "own_piece_attention_mass",
        "opponent_piece_attention_mass",
        "attended_coord_rank_mean",
        "attended_coord_file_mean",
        "attended_coord_rank_var",
        "attended_coord_file_var",
        "query_diversity",
        "attended_value_norm",
        "token_feature_energy",
        "side_to_move_white",
    ):
        assert output[key].shape == (2,)
        assert torch.isfinite(output[key]).all(), key
    for key in ("attention", "attended_values", "query_diagnostics"):
        assert torch.isfinite(output[key]).all(), key
    assert (output["attention_entropy_mean"] >= 0).all()
    assert (output["attention_entropy_mean"] <= 1).all()
    assert (output["attention_margin_mean"] >= 0).all()
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    tiny_config = {
        "input_channels": 18,
        "num_classes": 1,
        "token_dim": 16,
        "query_count": 6,
        "head_count": 4,
        "hidden_dim": 24,
        "head_hidden": 24,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "include_attention_diagnostics": True,
    }
    for ablation in (
        "none",
        "uniform_attention",
        "random_frozen_queries",
        "value_only_no_diagnostics",
        "diagnostics_only",
        "mean_pool_matched_params",
    ):
        tiny = build_model("set_query_attention_bottleneck", {**tiny_config, "ablation": ablation}).eval()
        with torch.no_grad():
            tiny_output = tiny(x)
        assert tiny_output["logits"].shape == (2,), ablation
        assert tiny_output["attention"].shape == (2, 6, 64), ablation
        assert torch.isfinite(tiny_output["logits"]).all(), ablation
        assert torch.allclose(tiny_output["attention"].sum(dim=-1), torch.ones(2, 6), atol=1.0e-5), ablation

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i102"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i161_multiplicative_conjunction_convnet_is_bespoke_and_conformant():
    folder = Path("ideas/i161_multiplicative_conjunction_convnet")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 3, 4, 4] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 8, 0, 4] = 1.0
    x[:, 10, 3, 4] = 1.0
    x[:, 11, 0, 7] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["product_norm_by_layer"].shape == (2, config["model"]["depth"])
    assert output["raw_product_norm_by_layer"].shape == (2, config["model"]["depth"])
    assert output["gate_mean_by_layer"].shape == (2, config["model"]["depth"])
    assert output["gate_saturation_by_layer"].shape == (2, config["model"]["depth"])
    assert {
        "product_branch_norm",
        "raw_product_branch_norm",
        "gate_mean",
        "gate_saturation",
        "branch_balance",
        "fusion_energy",
        "feature_energy",
        "aggregate_feature_energy",
    }.issubset(output)
    assert (output["raw_product_branch_norm"] > 0).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    additive = build_model(
        model_name,
        {
            "input_channels": 18,
            "num_classes": 1,
            "width": 16,
            "depth": 2,
            "branch_width": 8,
            "hidden_dim": 24,
            "dropout": 0.0,
            "use_batchnorm": False,
            "use_coordinate_planes": True,
            "ablation": "additive_only",
        },
    ).eval()
    with torch.no_grad():
        additive_output = additive(x)
    assert additive_output["logits"].shape == (2,)
    assert torch.isfinite(additive_output["product_branch_norm"]).all()

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i161"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i162_empty_square_opportunity_network_is_bespoke_and_conformant():
    folder = Path("ideas/i162_empty_square_opportunity_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 3, 4, 4] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 8, 0, 4] = 1.0
    x[:, 10, 3, 4] = 1.0
    x[:, 11, 0, 7] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["opportunity_maps"].shape == (2, config["model"]["opportunity_channels"], 8, 8)
    assert {
        "empty_opportunity_norm",
        "occupied_branch_norm",
        "empty_branch_norm",
        "occ_empty_interaction_energy",
        "occ_empty_gap",
        "opportunity_top_value",
        "top_opportunity_square",
        "occupancy_count",
        "empty_count",
        "occupancy_fraction",
        "aggregate_feature_energy",
        "escape_like_opportunity",
        "landing_like_opportunity",
        "blocker_like_opportunity",
        "promotion_lane_like_opportunity",
        "king_zone_empty_like_opportunity",
        "prob",
    }.issubset(output)
    assert (output["empty_count"] > 0).all()
    assert (output["occupancy_count"] > 0).all()
    assert (output["top_opportunity_square"] >= 0).all()
    assert (output["top_opportunity_square"] < 64).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    occupied_only = build_model(
        model_name,
        {
            "input_channels": 18,
            "num_classes": 1,
            "trunk_width": 16,
            "branch_width": 12,
            "opportunity_channels": 4,
            "hidden_dim": 24,
            "fusion_width": 24,
            "depth": 2,
            "topk_squares": 2,
            "dropout": 0.0,
            "use_batchnorm": False,
            "use_coordinate_planes": True,
            "ablation": "occupied_only",
        },
    ).eval()
    with torch.no_grad():
        occupied_only_output = occupied_only(x)
    assert occupied_only_output["logits"].shape == (2,)
    assert torch.isfinite(occupied_only_output["logits"]).all()
    assert torch.allclose(occupied_only_output["opportunity_maps"], torch.zeros_like(occupied_only_output["opportunity_maps"]))

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i162"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i163_global_scratchpad_boardnet_is_bespoke_and_conformant():
    folder = Path("ideas/i163_global_scratchpad_boardnet")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 3, 4, 4] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 8, 0, 4] = 1.0
    x[:, 10, 3, 4] = 1.0
    x[:, 11, 0, 7] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    steps = int(config["model"]["scratchpad_steps"])
    slots = int(config["model"]["memory_slots"])
    memory_dim = int(config["model"]["memory_dim"])
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["memory_slots"].shape == (2, slots, memory_dim)
    assert output["initial_memory_slots"].shape == (2, slots, memory_dim)
    assert output["memory_slot_norm_by_step"].shape == (2, steps, slots)
    assert output["memory_update_norm_by_step"].shape == (2, steps)
    assert output["board_activation_change_by_step"].shape == (2, steps)
    assert {
        "final_memory_norm",
        "initial_memory_norm",
        "memory_slot_similarity",
        "board_feature_energy",
        "board_pool_energy",
        "scratchpad_steps_used",
        "memory_slot_count",
        "prob",
    }.issubset(output)
    assert (output["scratchpad_steps_used"] == steps).all()
    assert (output["memory_slot_count"] == slots).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registry_output["memory_slot_norm_by_step"].shape == (2, steps, slots)
    assert torch.isfinite(registry_output["logits"]).all()
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    no_broadcast = build_model(
        model_name,
        {
            "input_channels": 18,
            "num_classes": 1,
            "width": 16,
            "memory_slots": 3,
            "memory_dim": 12,
            "scratchpad_steps": 2,
            "hidden_dim": 24,
            "dropout": 0.0,
            "use_batchnorm": False,
            "use_coordinate_planes": True,
            "ablation": "no_broadcast",
        },
    ).eval()
    with torch.no_grad():
        no_broadcast_output = no_broadcast(x)
    assert no_broadcast_output["logits"].shape == (2,)
    assert no_broadcast_output["memory_slot_norm_by_step"].shape == (2, 2, 3)
    assert torch.isfinite(no_broadcast_output["logits"]).all()

    single_slot = build_model(
        model_name,
        {
            "input_channels": 18,
            "num_classes": 1,
            "width": 16,
            "memory_slots": 3,
            "memory_dim": 12,
            "scratchpad_steps": 2,
            "hidden_dim": 24,
            "dropout": 0.0,
            "use_batchnorm": False,
            "use_coordinate_planes": True,
            "ablation": "single_slot",
        },
    ).eval()
    with torch.no_grad():
        single_slot_output = single_slot(x)
    assert single_slot_output["memory_slots"].shape == (2, 1, 12)
    assert torch.all(single_slot_output["memory_slot_similarity"] == 0)

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i163"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i235_toda_isospectral_flow_network_is_bespoke_and_conformant():
    folder = Path("ideas/i235_toda_isospectral_flow_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    fen = "4k3/8/8/3q4/4R3/8/8/4K3 w - - 0 1"
    x = torch.from_numpy(fen_to_tensor(fen)).float().unsqueeze(0).repeat(2, 1, 1, 1)
    with torch.no_grad():
        output = model(x)

    model_cfg = config["model"]
    operator_dim = int(model_cfg.get("operator_dim", 12))
    manakov_order = int(model_cfg.get("manakov_order", 4))

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "diag_initial",
        "diag_final",
        "off_initial",
        "off_final",
        "sorting_score",
        "max_off_diag_decay",
        "mean_off_diag_decay",
        "slowest_off_diag",
        "spectral_gap_estimate",
        "manakov_drift",
        "manakov_drift_max",
        "operator_frobenius_norm",
    }.issubset(output)
    assert output["diag_initial"].shape == (2, operator_dim)
    assert output["diag_final"].shape == (2, operator_dim)
    assert output["off_initial"].shape == (2, operator_dim - 1)
    assert output["off_final"].shape == (2, operator_dim - 1)
    assert output["manakov_drift"].shape == (2, manakov_order - 1)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key
    assert (output["off_initial"] > 0).all()
    assert (output["off_final"] > 0).all()
    assert (output["spectral_gap_estimate"] >= 0).all()

    model_name = config["model"]["name"]
    assert model_name == "toda_isospectral_flow_network"
    assert model_name not in RESEARCH_PACKET_MODEL_NAMES

    registry_model = build_model(
        model_name,
        {
            "input_channels": 18,
            "num_classes": 1,
            "channels": 16,
            "depth": 1,
            "operator_dim": 6,
            "flow_steps": 3,
            "flow_dt": 0.05,
            "manakov_order": 3,
            "hidden_dim": 16,
            "head_hidden": 16,
            "dropout": 0.0,
            "use_batchnorm": False,
        },
    ).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registry_output["manakov_drift"].shape == (2, 2)
    assert torch.isfinite(registry_output["logits"]).all()

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i235"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


PROMOTED_BATCH4_BESPOKE_IDEAS = [
    ("i203", "ideas/i203_hierarchical_tactical_option_network", "hierarchical_tactical_option_network", {"option_utilities", "hierarchical_gate_entropy"}),
    ("i204", "ideas/i204_cross_defense_consistency_network", "cross_defense_consistency_network", {"cross_defense_agreement", "cross_defense_variance"}),
    ("i205", "ideas/i205_defender_timing_schedule_network", "defender_timing_schedule_network", {"schedule_overrun", "schedule_entropy"}),
    ("i206", "ideas/i206_discovered_ray_switchboard_network", "discovered_ray_switchboard_network", {"discovered_ray_energy", "switchboard_entropy"}),
    ("i207", "ideas/i207_counterplay_insolvency_ledger", "counterplay_insolvency_ledger", {"insolvency_score", "ledger_balance"}),
    ("i208", "ideas/i208_pinned_mobility_nullspace_network", "pinned_mobility_nullspace_network", {"pin_nullspace_energy", "pin_nullspace_ratio"}),
    ("i209", "ideas/i209_tactical_effective_resistance_network", "tactical_effective_resistance_network", {"effective_resistance", "potential_spread"}),
    ("i210", "ideas/i210_defender_opportunity_cost_auction_network", "defender_opportunity_cost_auction_network", {"shadow_prices", "auction_total_value"}),
    ("i211", "ideas/i211_role_counterfactual_necessity_network", "role_counterfactual_necessity_network", {"role_necessity", "max_role_necessity"}),
    ("i212", "ideas/i212_phase_specialist_calibration_mixture", "phase_specialist_calibration_mixture", {"phase_probs", "expert_logits"}),
    ("i213", "ideas/i213_forced_target_funnel_network", "forced_target_funnel_network", {"funnel_concentration", "funnel_entropy"}),
    ("i214", "ideas/i214_tactical_subgoal_automaton_network", "tactical_subgoal_automaton_network", {"automaton_terminal_state", "transition_entropy"}),
    ("i215", "ideas/i215_masked_codec_interaction_curvature_network", "masked_codec_interaction_curvature_network", {"interaction_curvature", "reconstruction_error_low"}),
    ("i216", "ideas/i216_non_puzzle_score_curl_divergence_bottleneck", "non_puzzle_score_curl_divergence_bottleneck", {"score_curl_mean", "score_divergence_mean"}),
    ("i217", "ideas/i217_ray_grammar_edit_distance_network", "ray_grammar_edit_distance_network", {"global_min_edit_distance", "min_edit_distance_per_template"}),
]


@pytest.mark.parametrize("idea_id,folder_path,model_name,expected_keys", PROMOTED_BATCH4_BESPOKE_IDEAS)
def test_batch4_promoted_idea_is_bespoke_and_conformant(idea_id, folder_path, model_name, expected_keys):
    folder = Path(folder_path)
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert expected_keys.issubset(output.keys())
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == model_name
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == idea_id]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


PROMOTED_LA_VARIANT_BESPOKE_IDEAS = [
    (
        "i218",
        "ideas/i218_orbit_disagreement_residual_network",
        "orbit_disagreement_residual_network",
        {"orbit_residual_mean_norm", "view_logit_disagreement"},
    ),
    (
        "i219",
        "ideas/i219_hall_defect_dual_residual_network",
        "hall_defect_dual_residual_network",
        {"primal_violation_final", "dual_norm_final"},
    ),
    (
        "i220",
        "ideas/i220_credal_temperature_field_network",
        "credal_temperature_field_network",
        {"credal_temperature", "credal_smoothing"},
    ),
    (
        "i221",
        "ideas/i221_sylvester_tactical_coupling_network",
        "sylvester_tactical_coupling_network",
        {"sylvester_frobenius", "sylvester_resonance_min"},
    ),
    (
        "i222",
        "ideas/i222_schur_complement_defender_network",
        "schur_complement_defender_network",
        {"schur_inertia_neg", "schur_log_det_S"},
    ),
    (
        "i223",
        "ideas/i223_bures_wasserstein_threat_network",
        "bures_wasserstein_threat_network",
        {"bures_distance_gap", "bures_log_det_sigma"},
    ),
    (
        "i224",
        "ideas/i224_numerical_range_boundary_network",
        "numerical_range_boundary_network",
        {"non_normality_gap", "boundary_support"},
    ),
    (
        "i225",
        "ideas/i225_lyapunov_threat_stability_network",
        "lyapunov_threat_stability_network",
        {"lyapunov_cond_P", "lyapunov_hurwitz_indicator"},
    ),
    (
        "i226",
        "ideas/i226_pfaffian_skew_threat_network",
        "pfaffian_skew_threat_network",
        {"pfaffian_signed_log", "pfaffian_sign_balance"},
    ),
    (
        "i227",
        "ideas/i227_padic_ultrametric_threat_network",
        "padic_ultrametric_threat_network",
        {"padic_depth_histogram", "padic_newton_slopes"},
    ),
    (
        "i228",
        "ideas/i228_free_probability_r_transform_network",
        "free_probability_r_transform_network",
        {"free_coupling_distance", "free_cumulant_mismatch"},
    ),
]


@pytest.mark.parametrize("idea_id,folder_path,model_name,expected_keys", PROMOTED_LA_VARIANT_BESPOKE_IDEAS)
def test_la_variant_promoted_idea_is_bespoke_and_conformant(idea_id, folder_path, model_name, expected_keys):
    folder = Path(folder_path)
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert expected_keys.issubset(output.keys())
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == model_name
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == idea_id]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i039_ray_language_automaton_network_is_bespoke_and_conformant():
    folder = Path("ideas/i039_ray_language_automaton_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0  # white king
    x[:, 11, 0, 4] = 1.0  # black king
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "ray_scores",
        "ray_language_energy",
        "ray_score_logsumexp",
        "ray_automaton_diversity",
        "ray_axis_max",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "ray_language_automaton_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i039"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i040_kinematic_commutator_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i040_kinematic_commutator_bottleneck_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0   # white king e1
    x[:, 11, 0, 4] = 1.0  # black king e8
    x[:, 3, 7, 0] = 1.0   # white rook a1
    x[:, 10, 0, 0] = 1.0  # black queen a8
    x[:, 1, 7, 1] = 1.0   # white knight b1
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "commutator_field",
        "pair_stats",
        "bracket_energy",
        "bracket_max",
        "commutator_field_energy",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "kinematic_commutator_bottleneck_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i040"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i041_centered_tempo_odd_interventional_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i041_centered_tempo_odd_interventional_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[0, 12] = 1.0  # white-to-move
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    x[1, 12] = 0.0  # black-to-move
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 1, 7, 1] = 1.0   # white knight b1
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "tempo_odd_norm",
        "tempo_even_norm",
        "null_odd_norm",
        "centered_odd_norm",
        "side_intervention_gap",
        "centered_odd_energy",
        "centered_odd",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Centered odd map must be anti-invariant under the side-to-move involution.
    x_tau = x.clone()
    x_tau[:, 12] = 1.0 - x[:, 12]
    with torch.no_grad():
        output_tau = model(x_tau)
    centered_odd_sum = output["centered_odd"] + output_tau["centered_odd"]
    assert centered_odd_sum.abs().max().item() < 1.0e-5

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "centered_tempo_odd_interventional_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i041"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i042_legal_automorphism_quotient_network_is_bespoke_and_conformant():
    folder = Path("ideas/i042_legal_automorphism_quotient_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # Sample 0: white-to-move, kings on e1/e8, white rook a1, black queen a8.
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[0, 3, 7, 0] = 1.0
    x[0, 10, 0, 0] = 1.0
    x[0, 13] = 1.0  # white kingside castling rights
    x[0, 16] = 1.0  # black queenside castling rights
    x[0, 17, 5, 3] = 1.0  # en-passant target square
    # Sample 1: black-to-move, no castling, no ep.
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[1, 1, 7, 1] = 1.0
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "z_invariant",
        "invariant_norm",
        "character_energy",
        "character_norms",
        "file_mirror_character_norm",
        "color_flip_character_norm",
        "joint_character_norm",
        "orbit_variance",
        "character_penalty",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key
    assert output["character_norms"].shape == (2, 3)

    # The invariant latent and logits are exactly invariant under the
    # four legal automorphisms.
    from chess_nn_playground.models.legal_automorphism_quotient_network import (
        LegalAutomorphismTransform,
    )

    transform = LegalAutomorphismTransform()
    x_m = transform.file_mirror(x)
    x_q = transform.color_flip(x)
    x_mq = transform.file_mirror(x_q)
    with torch.no_grad():
        output_m = model(x_m)
        output_q = model(x_q)
        output_mq = model(x_mq)
    for transformed_output in (output_m, output_q, output_mq):
        assert (output["logits"] - transformed_output["logits"]).abs().max().item() < 1.0e-5
        assert (output["z_invariant"] - transformed_output["z_invariant"]).abs().max().item() < 1.0e-5

    # Both involutions square to identity and commute (C2 x C2 structure).
    assert (transform.file_mirror(x_m) - x).abs().max().item() == 0.0
    assert (transform.color_flip(x_q) - x).abs().max().item() == 0.0
    assert (
        transform.file_mirror(transform.color_flip(x))
        - transform.color_flip(transform.file_mirror(x))
    ).abs().max().item() == 0.0

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "legal_automorphism_quotient_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i042"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i043_side_canonical_rule_partition_invariant_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i043_side_canonical_rule_partition_invariant_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # Sample 0: white-to-move, kings + a couple of pieces.
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    x[0, 13] = 1.0        # white kingside castling rights
    x[0, 17, 5, 3] = 1.0  # en-passant target square
    # Sample 1: black-to-move, distinct geometry.
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 1, 7, 1] = 1.0   # white knight b1
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "z",
        "mu",
        "logvar",
        "kl",
        "phase_logits",
        "adv_logits",
        "color_logits",
        "phase_labels",
        "adv_labels",
        "color_labels",
        "group_ids",
        "total_material",
        "side_relative_advantage",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Adversary head logits agree with the configured partition arity.
    assert output["phase_logits"].shape == (2, 3)
    assert output["adv_logits"].shape == (2, 5)
    assert output["color_logits"].shape == (2, 2)
    # Deterministic partitions stay inside their declared ranges.
    assert int(output["phase_labels"].min()) >= 0 and int(output["phase_labels"].max()) <= 2
    assert int(output["adv_labels"].min()) >= 0 and int(output["adv_labels"].max()) <= 4
    assert int(output["color_labels"].min()) >= 0 and int(output["color_labels"].max()) <= 1
    assert int(output["group_ids"].min()) >= 0 and int(output["group_ids"].max()) <= 29
    # Sample 0 is white-to-move, sample 1 is black-to-move (pre-canonical color).
    assert output["color_labels"].tolist() == [0, 1]

    # Side-to-move canonicalization: under white-to-move the canonical
    # tensor must equal the identity on piece/castling/ep planes
    # (with the absolute side-to-move plane removed). Under black-to-move
    # the canonicalizer must vertically flip the white-piece planes onto
    # the enemy slot.
    from chess_nn_playground.models.rule_partition_invariant_bottleneck import (
        Simple18SideCanonicalizer,
    )

    canon = Simple18SideCanonicalizer()
    canonical = canon(x)
    assert canonical.shape == (2, 17, 8, 8)
    # White-to-move: friendly piece slot equals original white planes.
    assert torch.equal(canonical[0, 0:6], x[0, 0:6])
    # Black-to-move: friendly slot equals vertically flipped black planes
    # (so the moving side is always "friendly").
    assert torch.equal(canonical[1, 0:6], torch.flip(x[1, 6:12], dims=[1]))

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "side_canonical_rule_partition_invariant_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i043"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i044_masked_board_code_length_surprise_network_is_bespoke_and_conformant():
    folder = Path("ideas/i044_masked_board_code_length_surprise_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # Sample 0: white-to-move with kings on e1/e8 and a few pieces.
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    # Sample 1: black-to-move with a different geometry.
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 1, 7, 1] = 1.0   # white knight b1
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "code_length_field",
        "code_length_scaled_field",
        "entropy_field",
        "p_true_field",
        "code_length_mean",
        "code_length_max",
        "entropy_mean",
        "entropy_max",
        "p_true_mean",
        "codec_nll",
        "mask_coverage",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key
    # Spatial diagnostic shapes.
    assert output["code_length_field"].shape == (2, 8, 8)
    assert output["entropy_field"].shape == (2, 8, 8)
    assert output["p_true_field"].shape == (2, 8, 8)
    # The 2x2 residue mask bank covers every square exactly once.
    assert torch.allclose(output["mask_coverage"], torch.ones(2, 8, 8))
    # P_true is a probability and code length is non-negative.
    assert (output["p_true_field"] >= 0.0).all()
    assert (output["p_true_field"] <= 1.0).all()
    assert (output["code_length_field"] >= 0.0).all()

    # Tokenizer round-trip on a clean board.
    from chess_nn_playground.models.masked_surprise_codec import (
        MaskBank2x2Residues,
        Simple18PieceTokenizer,
    )

    tokenizer = Simple18PieceTokenizer(strict=True)
    tokens = tokenizer(x)
    assert tokens.shape == (2, 8, 8)
    # White king on e1 -> WK token (channel 5, so token = 6).
    assert int(tokens[0, 7, 4]) == 6
    # Black queen on a8 -> BQ token (channel 10, so token = 11).
    assert int(tokens[0, 0, 0]) == 11
    # Empty squares stay at token 0.
    assert int(tokens[0, 4, 4]) == 0

    # The fixed mask bank covers every square exactly once.
    bank = MaskBank2x2Residues()
    masks = bank.get_masks(device=torch.device("cpu"), dtype=torch.float32)
    assert masks.shape == (4, 1, 8, 8)
    assert torch.allclose(masks.sum(dim=0), torch.ones(1, 8, 8))

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "masked_board_code_length_surprise_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i044"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i045_credal_near_puzzle_evidence_network_is_bespoke_and_conformant():
    folder = Path("ideas/i045_credal_near_puzzle_evidence_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # Sample 0: white-to-move with kings on e1/e8 and a few pieces.
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    # Sample 1: black-to-move with a different geometry.
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 1, 7, 1] = 1.0   # white knight b1
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "alpha",
        "alpha_pos",
        "alpha_neg",
        "evidence",
        "evidence_pos",
        "evidence_neg",
        "evidence_mass",
        "mu_pos",
        "uncertainty",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Dirichlet structural invariants from math_thesis.md.
    alpha = output["alpha"]
    assert alpha.shape == (2, 2)
    assert (alpha >= 1.0 - 1e-5).all(), "alpha must satisfy alpha = 1 + softplus(...)"
    s = output["evidence_mass"]
    assert torch.allclose(s, alpha.sum(dim=1), atol=1e-5)
    mu = output["mu_pos"]
    assert torch.allclose(mu, alpha[:, 1] / s, atol=1e-5)
    # The reported single binary logit's sigmoid equals the Dirichlet predictive mean.
    assert torch.allclose(torch.sigmoid(output["logits"]), mu, atol=1e-5)
    # Uncertainty diagnostic equals 2/S.
    assert torch.allclose(output["uncertainty"], 2.0 / s, atol=1e-5)

    # Fail-closed adapter rejects unknown channel counts.
    from chess_nn_playground.models.credal_near_puzzle_evidence import (
        CredalEvidencePuzzleNet,
        FailClosedBoardAdapter,
    )

    with pytest.raises(ValueError):
        FailClosedBoardAdapter(input_channels=37, hidden_channels=8, encoding=None)
    with pytest.raises(ValueError):
        FailClosedBoardAdapter(input_channels=18, hidden_channels=8, encoding="lc0_unknown")
    # Allowing unknown channels constructs the adapter without raising.
    FailClosedBoardAdapter(
        input_channels=37, hidden_channels=8, encoding=None, allow_unknown_channels=True
    )

    # num_classes=2 head returns log(alpha+eps) so softmax equals the Dirichlet mean.
    binary_pair = CredalEvidencePuzzleNet(
        input_channels=18,
        num_classes=2,
        hidden_channels=16,
        hidden_dim=32,
        num_res_blocks=2,
        encoding="simple_18",
    ).eval()
    with torch.no_grad():
        pair_out = binary_pair(x)
    assert pair_out["logits"].shape == (2, 2)
    pair_softmax = torch.softmax(pair_out["logits"], dim=1)
    pair_alpha = pair_out["alpha"]
    assert torch.allclose(pair_softmax, pair_alpha / pair_alpha.sum(dim=1, keepdim=True), atol=1e-5)

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "credal_near_puzzle_evidence_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i045"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i046_rule_exact_orbit_bottleneck_network_is_bespoke_and_conformant():
    folder = Path("ideas/i046_rule_exact_orbit_bottleneck_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # Sample 0: white-to-move with kings on e1/e8 and a few pieces.
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    # Sample 1: black-to-move with a different geometry.
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 1, 7, 1] = 1.0   # white knight b1
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "identity_view_logit",
        "transformed_view_logit",
        "view_logit_gap",
        "identity_probability",
        "transformed_probability",
        "mean_view_probability",
        "orbit_probability_gap",
        "symmetry_residual",
        "latent_orbit_variance",
        "mechanism_energy",
        "orbit_size",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Orbit size diagnostic equals |G| = 2 for the default color_flip group.
    assert torch.allclose(output["orbit_size"], torch.full((2,), 2.0))
    # The symmetry residual matches the absolute view-probability gap.
    assert torch.allclose(
        output["symmetry_residual"], output["orbit_probability_gap"], atol=1e-6
    )

    # Color-flip invariance: the pooled binary logit for kappa(x) must match the logit for x.
    from chess_nn_playground.models.rule_exact_orbit_bottleneck import (
        RuleExactOrbitBottleneckNet,
        Simple18ColorFlipAdapter,
    )

    flipped = Simple18ColorFlipAdapter.color_flip(x)
    with torch.no_grad():
        output_flipped = model(flipped)
    assert torch.allclose(output["logits"], output_flipped["logits"], atol=1e-5)
    assert torch.allclose(
        output["mean_view_probability"], output_flipped["mean_view_probability"], atol=1e-5
    )

    # rank_flip_no_color falsifier shares parameters and runs end-to-end.
    falsifier = RuleExactOrbitBottleneckNet(
        input_channels=input_channels,
        num_classes=1,
        orbit_group="rank_flip_no_color",
        stem_width=int(config["model"].get("channels", 48)),
        latent_dim=int(config["model"].get("hidden_dim", 128)),
        num_blocks=int(config["model"].get("depth", 3)),
    ).eval()
    with torch.no_grad():
        falsifier_out = falsifier(x)
    assert falsifier_out["logits"].shape == (2,)

    # Identity orbit reduces to a single-view classifier (orbit_size = 1).
    identity_model = RuleExactOrbitBottleneckNet(
        input_channels=input_channels,
        num_classes=1,
        orbit_group="identity",
        stem_width=int(config["model"].get("channels", 48)),
        latent_dim=int(config["model"].get("hidden_dim", 128)),
        num_blocks=int(config["model"].get("depth", 3)),
    ).eval()
    with torch.no_grad():
        identity_out = identity_model(x)
    assert torch.allclose(identity_out["orbit_size"], torch.ones(2))

    # Fail-closed adapter rejects unknown channel counts unless explicitly opted out.
    adapter = Simple18ColorFlipAdapter()
    with pytest.raises(ValueError):
        adapter.make_orbit(torch.zeros(1, 17, 8, 8))
    Simple18ColorFlipAdapter(fail_closed_unknown_channels=False).make_orbit(
        torch.zeros(1, 17, 8, 8)
    )

    # num_classes=2 head returns log-probability vectors.
    pair = RuleExactOrbitBottleneckNet(
        input_channels=input_channels,
        num_classes=2,
        stem_width=16,
        latent_dim=24,
        num_blocks=2,
    ).eval()
    with torch.no_grad():
        pair_out = pair(x)
    assert pair_out["logits"].shape == (2, 2)

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "rule_exact_orbit_bottleneck_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i046"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i047_color_flip_orbit_evidence_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i047_color_flip_orbit_evidence_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # Sample 0: white-to-move with kings on e1/e8 and a few pieces.
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    # Sample 1: black-to-move with a different geometry.
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 1, 7, 1] = 1.0   # white knight b1
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "negative_evidence",
        "puzzle_evidence",
        "evidence_balance",
        "view_negative_evidence_gap",
        "view_puzzle_evidence_gap",
        "orbit_evidence_residual",
        "symmetry_residual",
        "latent_orbit_variance",
        "identity_puzzle_evidence",
        "flipped_puzzle_evidence",
        "identity_negative_evidence",
        "flipped_negative_evidence",
        "intersection_energy",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Color-flip invariance: the binary logit for tau(x) must match the logit for x.
    from chess_nn_playground.models.color_flip_orbit_evidence import (
        ColorFlipOrbitAdapter,
        ColorFlipOrbitEvidenceNet,
    )

    flipped = ColorFlipOrbitAdapter().color_flip(x)
    with torch.no_grad():
        output_flipped = model(flipped)
    assert torch.allclose(output["logits"], output_flipped["logits"], atol=1e-5)
    assert torch.allclose(
        output["puzzle_evidence"], output_flipped["puzzle_evidence"], atol=1e-5
    )
    assert torch.allclose(
        output["negative_evidence"], output_flipped["negative_evidence"], atol=1e-5
    )

    # tau(tau(x)) == x for the deterministic color-flip adapter.
    twice = ColorFlipOrbitAdapter().color_flip(flipped)
    assert torch.allclose(twice, x, atol=1e-6)

    # bad_rank_color falsifier shares parameters and runs end-to-end.
    falsifier = ColorFlipOrbitEvidenceNet(
        input_channels=input_channels,
        num_classes=1,
        hidden_channels=int(config["model"].get("channels", 64)),
        latent_dim=int(config["model"].get("hidden_dim", 96)),
        num_res_blocks=int(config["model"].get("depth", 2)),
        orbit_transform="bad_rank_color",
    ).eval()
    with torch.no_grad():
        falsifier_out = falsifier(x)
    assert falsifier_out["logits"].shape == (2,)

    # Identity orbit reduces to a duplicated-view ablation.
    identity_model = ColorFlipOrbitEvidenceNet(
        input_channels=input_channels,
        num_classes=1,
        hidden_channels=int(config["model"].get("channels", 64)),
        latent_dim=int(config["model"].get("hidden_dim", 96)),
        num_res_blocks=int(config["model"].get("depth", 2)),
        orbit_transform="identity",
    ).eval()
    with torch.no_grad():
        identity_out = identity_model(x)
    # With identity orbit the two views are equal, so per-class evidence gaps are 0.
    assert torch.allclose(identity_out["view_puzzle_evidence_gap"], torch.zeros(2), atol=1e-6)

    # Fail-closed adapter rejects unknown channel counts unless explicitly opted out.
    adapter = ColorFlipOrbitAdapter()
    with pytest.raises(ValueError):
        adapter.make_orbit(torch.zeros(1, 17, 8, 8))
    ColorFlipOrbitAdapter(fail_closed_unknown_channels=False).make_orbit(
        torch.zeros(1, 17, 8, 8)
    )

    # num_classes=2 head returns two-class log-probability vectors.
    pair = ColorFlipOrbitEvidenceNet(
        input_channels=input_channels,
        num_classes=2,
        hidden_channels=16,
        latent_dim=24,
        num_res_blocks=2,
    ).eval()
    with torch.no_grad():
        pair_out = pair(x)
    assert pair_out["logits"].shape == (2, 2)

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "color_flip_orbit_evidence_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i047"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i048_rule_automorphism_quotient_bottleneck_network_is_bespoke_and_conformant():
    folder = Path("ideas/i048_rule_automorphism_quotient_bottleneck_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # Sample 0: white-to-move, kings on e1/e8, white rook a1, black queen a8,
    # castling rights present (so file mirror H must be invalid for this sample).
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    x[0, 13] = 1.0        # white kingside castling rights
    x[0, 16] = 1.0        # black queenside castling rights
    # Sample 1: black-to-move, no castling rights, no en-passant, kingside kings.
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 1, 7, 1] = 1.0   # white knight b1

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "valid_view_count",
        "file_mirror_valid",
        "orbit_variance",
        "masked_orbit_variance",
        "view_logit_variance",
        "symmetry_residual",
        "orbit_consistency",
        "reynolds_norm",
        "projection_norm",
        "mechanism_energy",
        "risk_variance_proxy",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Castling-gated file mirror: sample 0 has castling rights so H/HC must be
    # masked out (only |G_x|=2 valid views), sample 1 has no castling so all
    # four views are valid.
    assert torch.allclose(output["valid_view_count"], torch.tensor([2.0, 4.0]))
    assert torch.allclose(output["file_mirror_valid"], torch.tensor([0.0, 1.0]))

    # Color/turn reversal must be a valid orbit member, so the binary logit
    # must be exactly invariant under T_C on every sample.
    from chess_nn_playground.models.rule_automorphism_quotient import (
        RuleAutomorphismQuotientNet,
        Simple18AutomorphismOrbit,
    )

    adapter = Simple18AutomorphismOrbit()
    x_color = adapter.color_turn_reversal(x)
    with torch.no_grad():
        output_color = model(x_color)
    assert torch.allclose(output["logits"], output_color["logits"], atol=1e-5)
    # File mirror is exact only when castling is absent, which is true for
    # sample 1; it must not be assumed exact for sample 0.
    x_file = adapter.file_mirror(x)
    with torch.no_grad():
        output_file = model(x_file)
    assert torch.allclose(
        output["logits"][1:2], output_file["logits"][1:2], atol=1e-5
    )

    # Both legal involutions square to identity.
    assert torch.allclose(adapter.color_turn_reversal(x_color), x, atol=1e-6)
    assert torch.allclose(adapter.file_mirror(x_file), x, atol=1e-6)

    # Auxiliary path exposes the orbit consistency / VICReg / REx losses
    # required by the math thesis objective.
    with torch.no_grad():
        aux = model(x, return_aux=True)
    aux_keys = {
        "orbit_mask",
        "z",
        "projection",
        "view_logits",
        "view_two_class_logits",
        "orbit_consistency_loss",
        "vicreg_variance_loss",
        "vicreg_covariance_loss",
        "latent_small_loss",
    }
    assert aux_keys.issubset(aux.keys())
    assert aux["orbit_mask"].shape == (2, 4)
    assert aux["projection"].shape[0:2] == (2, 4)

    # Pseudo-orbit falsifier shares parameters and runs end-to-end while
    # marking every view as valid (semantics-destroying same-count control).
    falsifier = RuleAutomorphismQuotientNet(
        input_channels=input_channels,
        num_classes=1,
        hidden_channels=int(config["model"].get("channels", 64)),
        latent_dim=int(config["model"].get("hidden_dim", 96)),
        num_res_blocks=int(config["model"].get("depth", 2)),
        pseudo_orbit=True,
    ).eval()
    with torch.no_grad():
        falsifier_out = falsifier(x)
    assert falsifier_out["logits"].shape == (2,)
    assert torch.allclose(falsifier_out["valid_view_count"], torch.tensor([4.0, 4.0]))

    # Color/turn-only ablation reduces the orbit to {I, C}.
    color_only = RuleAutomorphismQuotientNet(
        input_channels=input_channels,
        num_classes=1,
        hidden_channels=int(config["model"].get("channels", 64)),
        latent_dim=int(config["model"].get("hidden_dim", 96)),
        num_res_blocks=int(config["model"].get("depth", 2)),
        use_file_mirror_if_castling_absent=False,
    ).eval()
    with torch.no_grad():
        color_only_out = color_only(x)
    assert torch.allclose(color_only_out["valid_view_count"], torch.tensor([2.0, 2.0]))

    # Fail-closed adapter rejects unknown channel counts unless explicitly opted out.
    with pytest.raises(ValueError):
        Simple18AutomorphismOrbit()(torch.zeros(1, 17, 8, 8))
    Simple18AutomorphismOrbit(fail_closed_unknown_channels=False)(
        torch.zeros(1, 17, 8, 8)
    )

    # num_classes=2 head returns two-class logits.
    pair = RuleAutomorphismQuotientNet(
        input_channels=input_channels,
        num_classes=2,
        hidden_channels=16,
        latent_dim=24,
        num_res_blocks=2,
    ).eval()
    with torch.no_grad():
        pair_out = pair(x)
    assert pair_out["logits"].shape == (2, 2)

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "rule_automorphism_quotient_bottleneck_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i048"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i049_tempo_odd_bottleneck_network_is_bespoke_and_conformant():
    folder = Path("ideas/i049_tempo_odd_bottleneck_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[0, 12] = 1.0  # white-to-move
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    x[1, 12] = 0.0  # black-to-move
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 1, 7, 1] = 1.0   # white knight b1
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "tempo_odd_norm",
        "tempo_even_norm",
        "odd_energy",
        "even_energy",
        "odd_to_even_energy_ratio",
        "side_intervention_gap",
        "odd_variance_loss",
        "en_passant_removed",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Walsh odd projection is anti-invariant under tau (side-to-move toggle).
    x_tau = x.clone()
    x_tau[:, 12] = 1.0 - x[:, 12]
    with torch.no_grad():
        aux = model(x, return_aux=True)
        aux_tau = model(x_tau, return_aux=True)
    odd_sum = aux["z_odd"] + aux_tau["z_odd"]
    even_diff = aux["z_even"] - aux_tau["z_even"]
    assert odd_sum.abs().max().item() < 1.0e-5
    assert even_diff.abs().max().item() < 1.0e-5

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "tempo_odd_bottleneck_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i049"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i050_king_anchored_euler_interaction_network_is_bespoke_and_conformant():
    folder = Path("ideas/i050_king_anchored_euler_interaction_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.king_anchored_euler_interaction_network import (
        KingAnchoredEulerInteractionNet,
        _cubical_chi,
    )

    assert isinstance(model, KingAnchoredEulerInteractionNet)
    assert not isinstance(model, ResearchPacketProbe)

    # cubical chi sanity checks for V - E + F on the 8x8 grid.
    single = torch.zeros(8, 8)
    single[3, 3] = 1.0
    assert _cubical_chi(single).item() == 1.0
    two_adj = torch.zeros(8, 8)
    two_adj[3, 3] = 1.0
    two_adj[3, 4] = 1.0
    assert _cubical_chi(two_adj).item() == 1.0
    two_sep = torch.zeros(8, 8)
    two_sep[0, 0] = 1.0
    two_sep[7, 7] = 1.0
    assert _cubical_chi(two_sep).item() == 2.0
    ring = torch.zeros(8, 8)
    for r, c in [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)]:
        ring[r, c] = 1.0
    assert _cubical_chi(ring).item() == 0.0

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[0, 12] = 1.0  # white to move
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 4, 7, 3] = 1.0   # white queen d1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    x[1, 12] = 0.0  # black to move
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 1, 7, 1] = 1.0   # white knight b1

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "role_curve_energy",
        "interaction_curve_energy",
        "opp_king_interaction_pressure",
        "own_king_interaction_pressure",
        "center_interaction_pressure",
        "own_role_count",
        "opp_role_count",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Side-relative role swap: a position with white to move and a position obtained by
    # toggling the side-to-move bit and swapping the white<->black piece planes should
    # have identical role tensors and therefore identical Euler features.
    swapped = x.clone()
    swapped[0, 12] = 0.0  # toggle stm
    # swap white/black piece planes for the first sample (channels 0..5 <-> 6..11)
    swapped[0, 0:6] = x[0, 6:12]
    swapped[0, 6:12] = x[0, 0:6]
    with torch.no_grad():
        aux_orig = model(x[:1], return_aux=True)
        aux_swap = model(swapped[:1], return_aux=True)
    assert torch.allclose(aux_orig["roles"], aux_swap["roles"])
    assert torch.allclose(aux_orig["features"], aux_swap["features"])

    # Euler additivity identity: J = chi(union) - chi(r) - chi(s) = -chi(intersection).
    # For distinct role bitboards which never share a 2-cell, this should equal
    # -chi(closure(shared boundary cells)). Verify that on a simple two-cell pair
    # (white king e1 + black queen e2 are not sharing a cell, but adjacent so their
    # cubical closures share a single grid edge → chi(intersection) = 1, J = -1).
    probe = torch.zeros(1, 18, 8, 8)
    probe[0, 12] = 1.0
    probe[0, 5, 7, 4] = 1.0   # white king e1 (own_king)
    probe[0, 10, 6, 4] = 1.0  # black queen e2 (opp_heavy, adjacent to own_king)
    probe[0, 11, 0, 4] = 1.0  # black king e8 anywhere
    with torch.no_grad():
        aux = model(probe, return_aux=True)
    role_curves = aux["role_curves"]  # (1, R, A, U, T)
    interaction_curves = aux["interaction_curves"]  # (1, P, A, U, T)

    from chess_nn_playground.models.king_anchored_euler_interaction_network import (
        DEFAULT_INTERACTION_PAIRS,
        ROLE_OPP_HEAVY,
        ROLE_OWN_KING,
    )

    pair_index = DEFAULT_INTERACTION_PAIRS.index((ROLE_OPP_HEAVY, ROLE_OWN_KING))
    # Pick the centre anchor (a=2) and direction +inf-style threshold (last τ) so the
    # half-plane swallows the whole board: J should equal chi(union) - chi(r) - chi(s).
    a, u, t = 2, 0, role_curves.shape[-1] - 1
    chi_r = role_curves[0, ROLE_OPP_HEAVY, a, u, t].item()
    chi_s = role_curves[0, ROLE_OWN_KING, a, u, t].item()
    union_role_mask = torch.clamp(
        aux["roles"][0, ROLE_OPP_HEAVY] + aux["roles"][0, ROLE_OWN_KING], 0.0, 1.0
    )
    chi_union = _cubical_chi(union_role_mask).item()
    expected_j = chi_union - chi_r - chi_s
    actual_j = interaction_curves[0, pair_index, a, u, t].item()
    assert abs(actual_j - expected_j) < 1.0e-5
    # Two adjacent unit cells of distinct roles: union chi = 1, individual chi = 1 each, J = -1.
    assert abs(actual_j - (-1.0)) < 1.0e-5

    # Registry-built model from the same config produces identical structure.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "king_anchored_euler_interaction_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i050"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i056_non_puzzle_score_field_bottleneck_network_is_bespoke_and_conformant():
    folder = Path("ideas/i056_non_puzzle_score_field_bottleneck_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.non_puzzle_score_field_bottleneck import (
        NonPuzzleScoreFieldBottleneckNetwork,
    )

    assert isinstance(model, NonPuzzleScoreFieldBottleneckNetwork)
    assert not isinstance(model, ResearchPacketProbe)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # white to move
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 0, 6, 4] = 1.0   # white pawn e2
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "score_field_norm",
        "score_residual_energy",
        "score_bottleneck_energy",
        "score_field_mean_abs",
        "score_field_max_abs",
        "score_per_sigma_norm",
        "recon_residual_l2",
        "mechanism_energy",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()
    assert output["score_per_sigma_norm"].shape == (2, model.num_noise_levels)
    assert output["recon_residual_l2"].shape == (2, model.num_noise_levels)
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Tweedie/Vincent identity: forward_with_aux exposes the raw score stack and
    # it must equal (recon - x) / sigma^2. We probe one sigma here.
    with torch.no_grad():
        aux = model.forward_with_aux(x)
    score_maps = aux["score_maps_per_sigma"]  # (B, K, C, 8, 8)
    assert score_maps.shape == (2, model.num_noise_levels, input_channels, 8, 8)
    sigma0 = model.noise_sigmas[0]
    with torch.no_grad():
        recon0 = model.denoiser(x, model._sigma_buffer.to(x.dtype)[0])
        expected_score0 = (recon0 - x) / (sigma0 * sigma0)
    assert torch.allclose(score_maps[:, 0], expected_score0, atol=1.0e-5)

    # Score-matching helper filters to class-0 rows when binary_label is provided.
    g = torch.Generator().manual_seed(0)
    label_all_one = torch.ones(2, dtype=torch.long)
    loss_no_zeros = model.denoising_score_matching_loss(x, binary_label=label_all_one, generator=g)
    assert torch.allclose(loss_no_zeros, loss_no_zeros.new_zeros(()))
    label_mixed = torch.tensor([0, 1])
    loss_mixed = model.denoising_score_matching_loss(x, binary_label=label_mixed, generator=g)
    assert torch.isfinite(loss_mixed)
    assert loss_mixed > 0

    # Freezing the score prior turns off gradient flow on the denoiser.
    model.freeze_score_prior()
    assert not any(p.requires_grad for p in model.denoiser.parameters())
    model.unfreeze_score_prior()
    assert all(p.requires_grad for p in model.denoiser.parameters())

    # Fail-closed adapter for unknown encodings.
    with pytest.raises(ValueError):
        module.build_model_from_config({"model": {**config["model"], "input_channels": 12}})

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "non_puzzle_score_field_bottleneck_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, NonPuzzleScoreFieldBottleneckNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i056"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i057_soft_formal_concept_closure_network_is_bespoke_and_conformant():
    folder = Path("ideas/i057_soft_formal_concept_closure_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.soft_formal_concept_closure import (
        SoftFormalConceptClosureNet,
        Simple18BoardAdapter,
        RuleAttributeBuilder,
        SoftConceptClosureLayer,
        ConceptClosureReadout,
        _row_column_preserving_rewire,
    )

    assert isinstance(model, SoftFormalConceptClosureNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.adapter, Simple18BoardAdapter)
    assert isinstance(model.attribute_builder, RuleAttributeBuilder)
    assert isinstance(model.closure, SoftConceptClosureLayer)
    assert isinstance(model.readout, ConceptClosureReadout)

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # white to move
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 0, 6, 4] = 1.0   # white pawn e2
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[1, 5, 7, 6] = 1.0
    x[1, 11, 0, 6] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "extent_mass_mean",
        "extent_mass_max",
        "closure_mass_mean",
        "closure_mass_max",
        "closure_expansion_l1_mean",
        "closure_violation_l1_mean",
        "closure_energy",
        "mechanism_energy",
        "intent_density_mean",
        "global_features",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # The soft Galois closure operates on the rule-attribute incidence matrix; the
    # builder must produce a binary [B, 64, M] tensor whose value space is {0, 1}.
    parsed = model.adapter(x)
    incidence, globals_tensor = model.attribute_builder(parsed)
    assert incidence.shape == (2, 64, model.attribute_builder.num_attributes)
    assert torch.all((incidence == 0.0) | (incidence == 1.0))
    assert globals_tensor.shape == (2, model.attribute_builder.num_globals)

    closure = model.closure(incidence)
    assert closure["intents"].shape == (model.num_concepts, model.attribute_builder.num_attributes)
    assert closure["extent"].shape == (2, model.num_concepts, 64)
    assert closure["closed_intent"].shape == (2, model.num_concepts, model.attribute_builder.num_attributes)

    # Row/column-preserving rewire must preserve row sums and column sums (Section 9 falsifier).
    g = torch.Generator().manual_seed(0)
    rewired = _row_column_preserving_rewire(incidence, generator=g, swap_steps=8)
    assert rewired.shape == incidence.shape
    assert torch.allclose(rewired.sum(dim=2), incidence.sum(dim=2))
    assert torch.allclose(rewired.sum(dim=1), incidence.sum(dim=1))

    # Fail-closed adapter for unknown encodings.
    with pytest.raises(ValueError):
        module.build_model_from_config({"model": {**config["model"], "input_channels": 12}})
    with pytest.raises(ValueError):
        module.build_model_from_config({"model": {**config["model"], "adapter": "lc0_static_112"}})

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "soft_formal_concept_closure_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, SoftFormalConceptClosureNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i057"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i058_determinantal_tactical_volume_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i058_determinantal_tactical_volume_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.determinantal_volume import (
        DeterminantalTacticalVolumeNet,
        Simple18OccupiedTokenExtractor,
        PieceSquareTokenEncoder,
        RoleGatedPSDVolume,
        DeterminantalVolumeHead,
    )

    assert isinstance(model, DeterminantalTacticalVolumeNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.token_extractor, Simple18OccupiedTokenExtractor)
    assert isinstance(model.token_encoder, PieceSquareTokenEncoder)
    assert isinstance(model.volume, RoleGatedPSDVolume)
    assert isinstance(model.head, DeterminantalVolumeHead)

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # white to move
    x[0, 5, 7, 4] = 1.0    # white king e1
    x[0, 11, 0, 4] = 1.0   # black king e8
    x[0, 0, 6, 4] = 1.0    # white pawn e2
    x[0, 3, 7, 0] = 1.0    # white rook a1
    x[1, 5, 7, 6] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[1, 1, 5, 5] = 1.0    # white knight f3

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "log_volume",
        "log_volume_mean",
        "log_volume_max",
        "log_volume_min",
        "trace",
        "trace_mean",
        "gate_mass",
        "gate_mass_mean",
        "top_eig_ratio",
        "top_eig_ratio_mean",
        "active_count",
        "mechanism_energy",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Token extractor: at most 32 occupied tokens, mask matches actual occupancy.
    tokens = model.token_extractor(x)
    assert tokens.features.shape == (2, model.max_tokens, model.token_extractor.feature_dim)
    assert tokens.mask.shape == (2, model.max_tokens)
    # Sample 0 has 4 pieces, sample 1 has 3 pieces.
    assert int(tokens.mask[0].sum().item()) == 4
    assert int(tokens.mask[1].sum().item()) == 3
    # Padded slots must be exactly zero.
    assert torch.all(tokens.features * (1.0 - tokens.mask).unsqueeze(-1) == 0.0)

    # Permutation invariance of log-volume over occupied tokens (the central thesis).
    embed = model.token_encoder(tokens.features, tokens.mask)
    volume_a = model.volume(embed, tokens.mask).log_volume
    perm = torch.randperm(model.max_tokens)
    embed_perm = embed[:, perm, :]
    mask_perm = tokens.mask[:, perm]
    volume_b = model.volume(embed_perm, mask_perm).log_volume
    assert torch.allclose(volume_a, volume_b, atol=1.0e-4, rtol=1.0e-4)

    # Diagonal-trace ablation must produce a different signal than the determinant.
    abl_cfg = dict(config["model"])
    abl_cfg["ablation"] = "diagonal_trace_only"
    abl_cfg.pop("name", None)
    abl_model = module.build_determinantal_tactical_volume_bottleneck_from_config(abl_cfg).eval()
    with torch.no_grad():
        abl_out = abl_model(x)
    assert abl_out["logits"].shape == (2,)
    assert torch.isfinite(abl_out["logits"]).all()
    assert abl_model.volume.ablation == "diagonal_trace_only"

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "determinantal_tactical_volume_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, DeterminantalTacticalVolumeNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i058"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i059_harmonic_board_potential_network_is_bespoke_and_conformant():
    folder = Path("ideas/i059_harmonic_board_potential_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.harmonic_board_potential_network import (
        HarmonicBoardPotentialNet,
        Simple18ChargeEncoder,
        FixedBoardPoissonSolver,
        PotentialStatsPool,
        HarmonicPotentialHead,
        _build_grid_laplacian,
    )

    assert isinstance(model, HarmonicBoardPotentialNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.charge_encoder, Simple18ChargeEncoder)
    assert isinstance(model.solver, FixedBoardPoissonSolver)
    assert isinstance(model.stats_pool, PotentialStatsPool)
    assert isinstance(model.head, HarmonicPotentialHead)

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # white to move
    x[0, 5, 7, 4] = 1.0    # white king e1
    x[0, 11, 0, 4] = 1.0   # black king e8
    x[0, 0, 6, 4] = 1.0    # white pawn e2
    x[0, 3, 7, 0] = 1.0    # white rook a1
    x[1, 5, 7, 6] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[1, 1, 5, 5] = 1.0    # white knight f3

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "charge_potential_energy",
        "charge_potential_energy_mean",
        "dirichlet_energy",
        "dirichlet_energy_mean",
        "potential_mean",
        "potential_std",
        "potential_absmax",
        "boundary_flux",
        "king_us_potential",
        "king_them_potential",
        "charge_magnitude",
        "mechanism_energy",
        "ablation_active",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    # Per-(charge, lambda) tensors must have shape (B, K, L).
    k = int(config["model"]["charge_channels"])
    num_l = len(config["model"]["lambdas"])
    assert output["charge_potential_energy"].shape == (2, k, num_l)
    assert output["dirichlet_energy"].shape == (2, k, num_l)
    assert output["potential_mean"].shape == (2, k, num_l)
    assert output["king_us_potential"].shape == (2, k, num_l)
    assert output["king_them_potential"].shape == (2, k, num_l)

    # Green matrices are precomputed and non-trainable.
    green = model.solver.green_matrices
    assert tuple(green.shape) == (num_l, 64, 64)
    # The harmonic Green matrices must really invert the screened Laplacian.
    laplacian = _build_grid_laplacian(model.solver.boundary).to(green.dtype)
    eye64 = torch.eye(64, dtype=green.dtype)
    for idx, lam in enumerate(model.solver.lambdas):
        product = (laplacian + lam * eye64) @ model.solver.green_harmonic[idx]
        assert torch.allclose(product, eye64, atol=1.0e-3, rtol=1.0e-3), (idx, lam)

    # Variational identity: u^T (L + lambda I) u = u^T rho when u = G_l rho.
    rho = model.charge_encoder(x)
    u = model.solver(rho)
    rho_flat = rho.reshape(2, k, 64)
    u_flat = u.reshape(2, k, num_l, 64)
    rho_dot_u = (rho_flat.unsqueeze(2) * u_flat).sum(dim=-1)
    # u^T (L + lambda I) u
    Lu = torch.einsum("ij,bklj->bkli", laplacian.float(), u_flat)
    quad = (u_flat * Lu).sum(dim=-1)
    lam_tensor = torch.tensor(model.solver.lambdas, dtype=quad.dtype).view(1, 1, num_l)
    quad = quad + lam_tensor * (u_flat * u_flat).sum(dim=-1)
    assert torch.allclose(quad, rho_dot_u, atol=1.0e-3, rtol=1.0e-3)

    # All three central falsifier ablations build, run, and produce finite logits.
    for ablation in ("random_orthogonal_solver", "local_gaussian_solver", "charge_only_stats"):
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = module.build_harmonic_board_potential_network_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,)
        assert torch.isfinite(abl_out["logits"]).all()
        assert abl_model.solver.ablation == ablation
        if ablation == "charge_only_stats":
            # Solver must produce identically-zero potentials so the head only
            # sees charge moments and constant-zero potential statistics.
            assert torch.all(abl_model.solver(rho) == 0.0)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "harmonic_board_potential_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, HarmonicBoardPotentialNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i059"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i060_tropical_constraint_circuit_network_is_bespoke_and_conformant():
    folder = Path("ideas/i060_tropical_constraint_circuit_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.tropical_constraint_circuit_network import (
        TropicalConstraintCircuitNet,
        Simple18LiteralCostEncoder,
        TropicalClauseLayer,
        TropicalMarginPool,
        TropicalConstraintHead,
    )

    assert isinstance(model, TropicalConstraintCircuitNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.encoder, Simple18LiteralCostEncoder)
    assert isinstance(model.clause_layer, TropicalClauseLayer)
    assert isinstance(model.margin_pool, TropicalMarginPool)
    assert isinstance(model.head, TropicalConstraintHead)

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[0, 0, 6, 4] = 1.0
    x[0, 3, 7, 0] = 1.0
    x[1, 5, 7, 6] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[1, 1, 5, 5] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "monomial_costs",
        "clause_softmin_cost",
        "clause_value",
        "clause_margin",
        "clause_entropy",
        "clause_mean_cost",
        "clause_softmin_probabilities",
        "effective_monomials_per_clause",
        "active_literal_mass",
        "mechanism_energy",
        "ablation_active",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            assert torch.isfinite(value).all(), key

    k = int(config["model"]["clause_count"])
    m = int(config["model"]["monomials_per_clause"])
    assert output["monomial_costs"].shape == (2, k, m)
    assert output["clause_softmin_cost"].shape == (2, k)
    assert output["clause_margin"].shape == (2, k)
    assert output["clause_entropy"].shape == (2, k)
    assert output["clause_softmin_probabilities"].shape == (2, k, m)

    # Clause weights and biases must be nonnegative (so monotonicity in
    # literals holds).
    weights = model.clause_layer.clause_weights()
    assert weights.shape == (k, m, model.clause_layer.literal_dim)
    assert torch.all(weights >= 0)
    bias = torch.nn.functional.softplus(model.clause_layer.bias_raw)
    assert torch.all(bias >= 0)

    # Literal costs are nonnegative (softplus output) and the literal-cost
    # encoder consumes board + 4 coordinate planes.
    costs = model.encoder(x)
    assert costs.shape == (2, model.literal_channels, 8, 8)
    assert torch.all(costs >= 0)

    # Best-second margin must be nonnegative for all clauses.
    assert torch.all(output["clause_margin"] >= -1.0e-5)

    # Softmin probabilities must sum to 1 along the monomial axis.
    probs_sum = output["clause_softmin_probabilities"].sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1.0e-4)

    # Soft-min identity: -tau * logsumexp(-m / tau, dim=-1) == clause_softmin_cost.
    tau = model.effective_temperature
    expected_softmin = -tau * torch.logsumexp(-output["monomial_costs"] / tau, dim=-1)
    assert torch.allclose(expected_softmin, output["clause_softmin_cost"], atol=1.0e-5)

    # Effective monomial count is in [1, M].
    eff = output["effective_monomials_per_clause"]
    assert torch.all(eff >= 1.0 - 1.0e-4)
    assert torch.all(eff <= float(m) + 1.0e-4)

    # All five central falsifier ablations build, run, and produce finite logits.
    for ablation in (
        "sum_product_clause",
        "mean_literal_pool",
        "literal_square_shuffle",
        "high_temperature_softmin",
        "material_only_literals",
    ):
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = module.build_tropical_constraint_circuit_network_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,)
        assert torch.isfinite(abl_out["logits"]).all()
        assert abl_model.ablation == ablation
        if ablation == "high_temperature_softmin":
            # The effective temperature must actually be larger than the
            # configured temperature, so softmin approaches averaging.
            assert float(abl_model.effective_temperature) > float(abl_model.softmin_temperature)
        if ablation == "sum_product_clause":
            # The clause aggregator changes from the soft-min surrogate to
            # the soft-average surrogate. With the same monomial costs,
            # the clause_value must lie between the per-clause min and the
            # per-clause mean (any convex combination of monomial costs).
            mono = abl_out["monomial_costs"]
            mn = mono.amin(dim=-1)
            mean = mono.mean(dim=-1)
            tol = 1.0e-3 * mean.abs().clamp_min(1.0)
            value = abl_out["clause_value"]
            assert torch.all(value >= mn - tol)
            assert torch.all(value <= mean + tol)
        if ablation == "literal_square_shuffle":
            # The fixed permutation buffer must be a permutation of
            # range(64), i.e. each index appears exactly once.
            perm = abl_model.square_permutation
            assert perm.shape == (64,)
            assert torch.equal(torch.sort(perm).values, torch.arange(64))

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "tropical_constraint_circuit_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, TropicalConstraintCircuitNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i060"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i061_grassmannian_principal_angle_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i061_grassmannian_principal_angle_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.grassmannian_principal_angle_bottleneck import (
        GrassmannianPrincipalAngleNet,
        Simple18OccupiedTokenExtractor,
        PieceSquareTokenEncoder,
        RoleGatedCovarianceSubspaces,
        PrincipalAngleSpectrum,
        GrassmannianAngleHead,
    )

    assert isinstance(model, GrassmannianPrincipalAngleNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.token_extractor, Simple18OccupiedTokenExtractor)
    assert isinstance(model.token_encoder, PieceSquareTokenEncoder)
    assert isinstance(model.role_subspaces, RoleGatedCovarianceSubspaces)
    assert isinstance(model.angle_module, PrincipalAngleSpectrum)
    assert isinstance(model.head, GrassmannianAngleHead)

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18
    role_count = int(config["model"]["role_count"])
    subspace_dim = int(config["model"]["subspace_dim"])
    expected_pairs = role_count * (role_count - 1) // 2
    assert model.angle_module.num_pairs == expected_pairs

    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 1, 6, 3] = 1.0
    x[:, 3, 5, 2] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 9, 2, 5] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "principal_angle_cosines",
        "principal_angle_radians",
        "pair_min_angle",
        "pair_max_angle",
        "pair_mean_angle",
        "pair_entropy",
        "role_eigenvalues",
        "role_gate_mass",
        "active_token_count",
        "mean_pair_cosine",
        "mean_pair_angle",
        "pair_mean_angle_std",
        "mechanism_energy",
        "eigen_mass",
        "ablation_active",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output)
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()

    assert output["principal_angle_cosines"].shape == (2, expected_pairs, subspace_dim)
    assert output["principal_angle_radians"].shape == (2, expected_pairs, subspace_dim)
    assert output["role_eigenvalues"].shape == (2, role_count, subspace_dim)
    assert output["role_gate_mass"].shape == (2, role_count)

    cosines = output["principal_angle_cosines"]
    angles = output["principal_angle_radians"]
    assert torch.all(cosines >= 0.0 - 1.0e-5)
    assert torch.all(cosines <= 1.0 + 1.0e-5)
    assert torch.all(angles >= 0.0 - 1.0e-5)
    assert torch.all(angles <= torch.pi / 2 + 1.0e-3)
    # cosines are sorted descending, angles are sorted ascending.
    cos_diff = cosines[..., 1:] - cosines[..., :-1]
    assert torch.all(cos_diff <= 1.0e-5)
    angle_diff = angles[..., 1:] - angles[..., :-1]
    assert torch.all(angle_diff >= -1.0e-5)
    # arccos identity
    expected_angles = torch.arccos(cosines.clamp(0.0, 1.0 - 1.0e-6))
    assert torch.allclose(expected_angles, angles, atol=1.0e-3)

    # Eigenvalues sorted descending and nonneg.
    eigenvalues = output["role_eigenvalues"]
    eig_diff = eigenvalues[..., 1:] - eigenvalues[..., :-1]
    assert torch.all(eig_diff <= 1.0e-5)
    assert torch.all(eigenvalues >= 0.0)

    # Backward through the bespoke pipeline must be finite (eigh + svdvals
    # remain stable thanks to the linear-tilt regulariser inside
    # RoleGatedCovarianceSubspaces).
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    enc_grad = trainable.token_encoder.mlp[0].weight.grad
    gate_grad = trainable.role_subspaces.gate_mlp[0].weight.grad
    head_grad = trainable.head.mlp[1].weight.grad
    assert enc_grad is not None and torch.isfinite(enc_grad).all()
    assert gate_grad is not None and torch.isfinite(gate_grad).all()
    assert head_grad is not None and torch.isfinite(head_grad).all()

    # All section-9 falsifier ablations must build, run, and produce finite logits.
    for ablation in (
        "no_cross_angles",
        "batch_shuffled_angles",
        "eigenvalues_only",
        "pooled_token_head",
        "no_orthonormalization",
    ):
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = module.build_grassmannian_principal_angle_bottleneck_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation

    # Permutation invariance over occupied tokens: shuffling the order of
    # piece planes does not change which (channel, square) cells are
    # occupied, and the model's covariance is a sum so a deterministic
    # shuffle of the active-token order should not move the principal-angle
    # spectrum.
    perm = torch.tensor([5, 0, 1, 6, 2, 11, 9, 3, 4, 7, 8, 10])
    x_perm = x.clone()
    x_perm[:, :12] = x[:, perm]
    with torch.no_grad():
        output_perm = model(x_perm)
    # The set of (channel-mapped) occupied cells changes with this remap,
    # so we instead check that a *batch reorder* (which never touches
    # token order within a sample) leaves spectra unchanged.
    x_shuffle = x[[1, 0]]
    with torch.no_grad():
        out_shuffle = model(x_shuffle)
    assert torch.allclose(out_shuffle["logits"], output["logits"][[1, 0]], atol=1.0e-4)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "grassmannian_principal_angle_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, GrassmannianPrincipalAngleNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i061"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i062_matrix_pencil_generalized_spectrum_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i062_matrix_pencil_generalized_spectrum_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.matrix_pencil_generalized_spectrum_bottleneck import (
        MatrixPencilGeneralizedSpectrumNet,
        Simple18OccupiedTokenExtractor,
        PieceSquareTokenEncoder,
        LowRankBoardMatrixPair,
        GeneralizedSpectrumLayer,
        MatrixPencilHead,
    )

    assert isinstance(model, MatrixPencilGeneralizedSpectrumNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.token_extractor, Simple18OccupiedTokenExtractor)
    assert isinstance(model.token_encoder, PieceSquareTokenEncoder)
    assert isinstance(model.matrix_pair, LowRankBoardMatrixPair)
    assert isinstance(model.spectrum, GeneralizedSpectrumLayer)
    assert isinstance(model.head, MatrixPencilHead)

    input_channels = int(config["model"]["input_channels"])
    matrix_dim = int(config["model"]["matrix_dim"])
    factor_rank = int(config["model"]["factor_rank"])
    probe_count = int(config["model"]["probe_count"])
    assert input_channels == 18

    x = torch.zeros(2, input_channels, 8, 8)
    # Two distinct boards so the batch_shuffled_b ablation has structure to break.
    x[0, 0, 6, 4] = 1.0
    x[0, 1, 6, 3] = 1.0
    x[0, 3, 5, 2] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 6, 1, 4] = 1.0
    x[0, 9, 2, 5] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[1, 0, 5, 3] = 1.0
    x[1, 2, 4, 4] = 1.0
    x[1, 4, 6, 2] = 1.0
    x[1, 5, 7, 6] = 1.0
    x[1, 6, 2, 5] = 1.0
    x[1, 8, 1, 3] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "generalized_eigenvalues",
        "log_generalized_eigenvalues",
        "rayleigh_probes",
        "matrix_a",
        "matrix_b",
        "eigenvalues_a",
        "eigenvalues_b",
        "trace_a",
        "trace_b",
        "trace_ratio",
        "condition_b",
        "proportionality_diagnostic",
        "mechanism_energy",
        "active_token_count",
        "ablation_active",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output)
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()

    assert output["generalized_eigenvalues"].shape == (2, matrix_dim)
    assert output["log_generalized_eigenvalues"].shape == (2, matrix_dim)
    assert output["rayleigh_probes"].shape == (2, probe_count)
    assert output["matrix_a"].shape == (2, matrix_dim, matrix_dim)
    assert output["matrix_b"].shape == (2, matrix_dim, matrix_dim)
    assert output["eigenvalues_a"].shape == (2, matrix_dim)
    assert output["eigenvalues_b"].shape == (2, matrix_dim)

    # Generalized eigenvalues are positive (B is PD, A is PSD + eps*I) and sorted descending.
    gen = output["generalized_eigenvalues"]
    assert torch.all(gen > 0.0)
    gen_diff = gen[..., 1:] - gen[..., :-1]
    assert torch.all(gen_diff <= 1.0e-4)

    # Separate spectra are nonneg and sorted descending.
    eig_a = output["eigenvalues_a"]
    eig_b = output["eigenvalues_b"]
    assert torch.all(eig_a >= 0.0)
    assert torch.all(eig_b > 0.0)
    a_diff = eig_a[..., 1:] - eig_a[..., :-1]
    b_diff = eig_b[..., 1:] - eig_b[..., :-1]
    assert torch.all(a_diff <= 1.0e-4)
    assert torch.all(b_diff <= 1.0e-4)

    # Backward through Cholesky + eigvalsh stays finite for the trainable model.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    enc_grad = trainable.token_encoder.mlp[0].weight.grad
    pair_grad_a = trainable.matrix_pair.value_a.weight.grad
    pair_grad_b = trainable.matrix_pair.value_b.weight.grad
    head_grad = trainable.head.mlp[1].weight.grad
    probe_grad = trainable.spectrum.probes.grad
    assert enc_grad is not None and torch.isfinite(enc_grad).all()
    assert pair_grad_a is not None and torch.isfinite(pair_grad_a).all()
    assert pair_grad_b is not None and torch.isfinite(pair_grad_b).all()
    assert head_grad is not None and torch.isfinite(head_grad).all()
    assert probe_grad is not None and torch.isfinite(probe_grad).all()

    # All section-9 falsifier ablations must build, run, and produce finite logits.
    for ablation in (
        "separate_spectra_only",
        "trace_ratio_only",
        "batch_shuffled_b",
        "random_factors",
        "single_matrix_spectrum",
        "mean_pool_head",
        "material_only_tokens",
    ):
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = module.build_matrix_pencil_generalized_spectrum_bottleneck_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "random_factors":
            for parameter in abl_model.token_encoder.parameters():
                assert not parameter.requires_grad, ablation
            for parameter in abl_model.matrix_pair.parameters():
                assert not parameter.requires_grad, ablation

    # Permutation invariance over batch: reordering samples in the batch must
    # preserve per-sample logits (the model has no cross-sample interaction
    # in the default ablation).
    x_shuffle = x[[1, 0]]
    with torch.no_grad():
        out_shuffle = model(x_shuffle)
    assert torch.allclose(out_shuffle["logits"], output["logits"][[1, 0]], atol=1.0e-4)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "matrix_pencil_generalized_spectrum_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, MatrixPencilGeneralizedSpectrumNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i062"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i063_polar_procrustes_alignment_bottleneck_is_bespoke_and_conformant():
    folder = Path("ideas/i063_polar_procrustes_alignment_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.polar_procrustes_alignment_bottleneck import (
        PolarProcrustesAlignmentNet,
        Simple18OwnOpponentTokenExtractor,
        PieceSquareTokenEncoder,
        RoleMatrixPooler,
        PolarProcrustesLayer,
        PolarProcrustesHead,
    )

    assert isinstance(model, PolarProcrustesAlignmentNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.token_extractor, Simple18OwnOpponentTokenExtractor)
    assert isinstance(model.token_encoder, PieceSquareTokenEncoder)
    assert isinstance(model.role_pooler, RoleMatrixPooler)
    assert isinstance(model.procrustes, PolarProcrustesLayer)
    assert isinstance(model.head, PolarProcrustesHead)

    input_channels = int(config["model"]["input_channels"])
    token_dim = int(config["model"]["token_dim"])
    role_count = int(config["model"]["role_count"])
    matrix_space = str(config["model"]["matrix_space"])
    assert input_channels == 18
    expected_matrix_dim = token_dim if matrix_space == "embedding" else role_count
    assert model.matrix_dim == expected_matrix_dim

    x = torch.zeros(2, input_channels, 8, 8)
    x[0, 0, 6, 4] = 1.0
    x[0, 1, 6, 3] = 1.0
    x[0, 3, 5, 2] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 6, 1, 4] = 1.0
    x[0, 9, 2, 5] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[1, 0, 5, 3] = 1.0
    x[1, 2, 4, 4] = 1.0
    x[1, 4, 6, 2] = 1.0
    x[1, 5, 7, 6] = 1.0
    x[1, 6, 2, 5] = 1.0
    x[1, 8, 1, 3] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "cross_covariance",
        "orthogonal_alignment",
        "singular_values",
        "polar_strain_diagonal",
        "procrustes_residual",
        "identity_residual",
        "alignment_improvement",
        "per_role_residual",
        "nuclear_norm",
        "spectral_norm",
        "stable_rank",
        "x_norm",
        "y_norm",
        "x_singular_values",
        "y_singular_values",
        "own_role_mass",
        "opp_role_mass",
        "active_token_count",
        "mechanism_energy",
        "ablation_active",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output)
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()

    assert output["cross_covariance"].shape == (2, expected_matrix_dim, expected_matrix_dim)
    assert output["orthogonal_alignment"].shape == (2, expected_matrix_dim, expected_matrix_dim)
    assert output["singular_values"].shape == (2, expected_matrix_dim)
    assert output["polar_strain_diagonal"].shape == (2, expected_matrix_dim)
    assert output["per_role_residual"].shape == (2, role_count)
    assert output["x_singular_values"].shape == (2, role_count)
    assert output["y_singular_values"].shape == (2, role_count)
    assert output["own_role_mass"].shape == (2, role_count)
    assert output["opp_role_mass"].shape == (2, role_count)

    # Q* must be orthogonal: Q* Q*^T = I, up to numerical noise.
    q = output["orthogonal_alignment"]
    eye = torch.eye(expected_matrix_dim).expand(2, -1, -1)
    qqt = torch.matmul(q, q.transpose(-1, -2))
    assert torch.allclose(qqt, eye, atol=1.0e-3)

    # Singular values are sorted descending and nonneg.
    sigma = output["singular_values"]
    sigma_diff = sigma[..., 1:] - sigma[..., :-1]
    assert torch.all(sigma_diff <= 1.0e-4)
    assert torch.all(sigma >= -1.0e-5)

    # Procrustes residual <= identity residual (alignment improvement >= 0 up to eps).
    improvement = output["alignment_improvement"]
    assert torch.all(improvement >= -1.0e-3)

    # Per-role residual norms are nonneg.
    assert torch.all(output["per_role_residual"] >= 0.0)
    # Procrustes residual equals the L2 norm of per_role_residual (its definition).
    rebuilt = torch.linalg.vector_norm(output["per_role_residual"], dim=-1)
    assert torch.allclose(rebuilt, output["procrustes_residual"], atol=1.0e-4)

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    enc_grad = trainable.token_encoder.mlp[0].weight.grad
    own_query_grad = trainable.role_pooler.own_query.weight.grad
    opp_query_grad = trainable.role_pooler.opp_query.weight.grad
    head_grad = trainable.head.mlp[1].weight.grad
    assert enc_grad is not None and torch.isfinite(enc_grad).all()
    assert own_query_grad is not None and torch.isfinite(own_query_grad).all()
    assert opp_query_grad is not None and torch.isfinite(opp_query_grad).all()
    assert head_grad is not None and torch.isfinite(head_grad).all()

    # All section-9 falsifier ablations must build, run, and produce finite logits.
    for ablation in (
        "separate_matrix_stats_only",
        "identity_alignment_only",
        "random_orthogonal_alignment",
        "batch_shuffled_opponent",
        "material_only_matrices",
        "role_pool_mean_only",
        "singular_values_only",
    ):
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = module.build_polar_procrustes_alignment_bottleneck_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "identity_alignment_only":
            # Q* should be the identity for every sample.
            q_abl = abl_out["orthogonal_alignment"]
            target_eye = torch.eye(expected_matrix_dim).expand(2, -1, -1)
            assert torch.allclose(q_abl, target_eye, atol=1.0e-5), ablation

    # Permutation invariance over batch: reordering samples in the batch must
    # preserve per-sample logits (the model has no cross-sample interaction
    # in the default ablation).
    x_shuffle = x[[1, 0]]
    with torch.no_grad():
        out_shuffle = model(x_shuffle)
    assert torch.allclose(out_shuffle["logits"], output["logits"][[1, 0]], atol=1.0e-4)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "polar_procrustes_alignment_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, PolarProcrustesAlignmentNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i063"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i064_multi_scale_dilated_board_mixer_cnn_is_bespoke_and_conformant():
    folder = Path("ideas/i064_multi_scale_dilated_board_mixer_cnn")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.multi_scale_dilated_board_mixer_cnn import (
        BoardCoordinatePlanes,
        GlobalContextGate,
        MultiScaleBoardMixerCNN,
        MultiScaleDilatedMixerBlock,
        MultiScaleHead,
        build_multi_scale_dilated_board_mixer_cnn_from_config,
    )

    assert isinstance(model, MultiScaleBoardMixerCNN)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.coordinate_planes, BoardCoordinatePlanes)
    assert isinstance(model.global_gate, GlobalContextGate)
    assert isinstance(model.head, MultiScaleHead)
    assert all(isinstance(block, MultiScaleDilatedMixerBlock) for block in model.blocks)

    input_channels = int(config["model"]["input_channels"])
    width = int(config["model"]["channels"])
    num_blocks = int(config["model"]["depth"])
    assert input_channels == 18
    assert model.config.width == width
    assert model.config.num_blocks == num_blocks
    # Default: parallel d=1, d=2, d=3 and 1x1 branches.
    assert len(model.blocks) == num_blocks
    assert len(model.blocks[0].branches) == 4

    x = torch.zeros(2, input_channels, 8, 8)
    x[0, 0, 6, 4] = 1.0
    x[0, 1, 6, 3] = 1.0
    x[0, 3, 5, 2] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 6, 1, 4] = 1.0
    x[0, 9, 2, 5] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[1, 0, 5, 3] = 1.0
    x[1, 2, 4, 4] = 1.0
    x[1, 4, 6, 2] = 1.0
    x[1, 5, 7, 6] = 1.0
    x[1, 6, 2, 5] = 1.0
    x[1, 8, 1, 3] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "two_class_logits",
        "stem_energy",
        "trunk_energy",
        "coord_plane_energy",
        "context_gate_mean",
        "context_gate_std",
        "context_gate_min",
        "context_gate_max",
        "branch_count",
        "active_dilations",
        "ablation_active",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output)
    assert output["logits"].shape == (2,)
    assert output["two_class_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()

    # Coordinate plane append produces non-trivial energy.
    assert torch.all(output["coord_plane_energy"] > 0.0)
    # Sigmoid gate values must lie in (0, 1).
    assert torch.all(output["context_gate_min"] >= 0.0)
    assert torch.all(output["context_gate_max"] <= 1.0)
    assert torch.all(output["branch_count"] == 4.0)
    assert torch.all(output["active_dilations"] == 3.0)
    assert torch.all(output["ablation_active"] == 0.0)

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    stem_grad = trainable.stem[0].weight.grad
    branch_grad = trainable.blocks[0].branches[0][0].weight.grad
    project_grad = trainable.blocks[0].project.weight.grad
    gate_grad = trainable.global_gate.mlp[0].weight.grad
    head_grad = trainable.head.classifier[1].weight.grad
    assert stem_grad is not None and torch.isfinite(stem_grad).all()
    assert branch_grad is not None and torch.isfinite(branch_grad).all()
    assert project_grad is not None and torch.isfinite(project_grad).all()
    assert gate_grad is not None and torch.isfinite(gate_grad).all()
    assert head_grad is not None and torch.isfinite(head_grad).all()

    # Section-6 ablations build, run, and produce finite logits.
    for ablation in (
        "single_dilation_matched",
        "no_dilation_3",
        "no_coordinate_planes",
        "no_global_context_gate",
        "small_width_control",
        "residual_cnn_matched_params",
    ):
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = build_multi_scale_dilated_board_mixer_cnn_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "single_dilation_matched":
            assert len(abl_model.blocks[0].branches) == 1
        elif ablation == "no_dilation_3":
            assert len(abl_model.blocks[0].branches) == 3
        elif ablation == "no_coordinate_planes":
            assert abl_model.coordinate_planes is None
        elif ablation == "no_global_context_gate":
            assert abl_model.global_gate is None
        elif ablation == "small_width_control":
            assert abl_model.config.width == max(8, int(config["model"]["channels"]) // 2)
        elif ablation == "residual_cnn_matched_params":
            assert abl_model.residual_control is not None
            assert abl_model.head is None

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "multi_scale_dilated_board_mixer_cnn"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, MultiScaleBoardMixerCNN)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i064"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i065_piece_token_cnn_hybrid_is_bespoke_and_conformant():
    folder = Path("ideas/i065_piece_token_cnn_hybrid")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.piece_token_cnn_hybrid import (
        BoardCNNTrunk,
        CNNTokenFusionHead,
        PieceTokenCNNHybrid,
        PieceTokenMixer,
        Simple18PieceTokenExtractor,
        TokenMixerLayer,
        build_piece_token_cnn_hybrid_from_config,
    )

    assert isinstance(model, PieceTokenCNNHybrid)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.extractor, Simple18PieceTokenExtractor)
    assert isinstance(model.cnn, BoardCNNTrunk)
    assert isinstance(model.token_mixer, PieceTokenMixer)
    assert isinstance(model.head, CNNTokenFusionHead)
    assert all(isinstance(layer, TokenMixerLayer) for layer in model.token_mixer.layers)

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18

    x = torch.zeros(2, input_channels, 8, 8)
    x[0, 0, 6, 4] = 1.0
    x[0, 1, 6, 3] = 1.0
    x[0, 3, 5, 2] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 6, 1, 4] = 1.0
    x[0, 9, 2, 5] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[1, 0, 5, 3] = 1.0
    x[1, 2, 4, 4] = 1.0
    x[1, 4, 6, 2] = 1.0
    x[1, 5, 7, 6] = 1.0
    x[1, 6, 2, 5] = 1.0
    x[1, 8, 1, 3] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "token_count",
        "piece_count",
        "material_balance",
        "cnn_energy",
        "token_energy",
        "cnn_token_interaction",
        "token_coordinate_energy",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert torch.all(output["cnn_energy"] >= 0.0)
    assert torch.all(output["token_energy"] >= 0.0)
    assert torch.all(output["cnn_token_interaction"] >= 0.0)
    assert torch.all(output["token_coordinate_energy"] >= 0.0)
    # 7 occupied tokens per board in the synthetic batch above.
    assert torch.all(output["token_count"] == 7.0)

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    cnn_grad = trainable.cnn.blocks[0].weight.grad
    encoder_grad = trainable.token_mixer.encoder[0].weight.grad
    head_grad = trainable.head.classifier[1].weight.grad
    cnn_proj_grad = trainable.head.cnn_proj.weight.grad
    token_proj_grad = trainable.head.token_proj.weight.grad
    assert cnn_grad is not None and torch.isfinite(cnn_grad).all()
    assert encoder_grad is not None and torch.isfinite(encoder_grad).all()
    assert head_grad is not None and torch.isfinite(head_grad).all()
    assert cnn_proj_grad is not None and torch.isfinite(cnn_proj_grad).all()
    assert token_proj_grad is not None and torch.isfinite(token_proj_grad).all()

    # Falsifier ablations must build, run, and produce finite logits.
    for ablation in (
        "cnn_only_matched",
        "token_only",
        "no_interaction_fusion",
        "material_token_only",
        "shuffle_token_coordinates",
        "single_token_layer",
    ):
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = build_piece_token_cnn_hybrid_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation in {"cnn_only_matched", "no_interaction_fusion"}:
            assert abl_model.head.include_interaction is False
        if ablation == "single_token_layer":
            assert len(abl_model.token_mixer.layers) == 1

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "piece_token_cnn_hybrid"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, PieceTokenCNNHybrid)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i065"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i067_finite_field_character_sum_board_network_is_bespoke_and_conformant():
    folder = Path("ideas/i067_finite_field_character_sum_board_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.finite_field_character_sum import (
        CharacterProbeTable,
        CharacterSumHead,
        FiniteFieldCharacterFeatures,
        FiniteFieldCharacterSumBoardNetwork,
        Simple18FiniteFieldEncoder,
    )

    assert isinstance(model, FiniteFieldCharacterSumBoardNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.encoder, Simple18FiniteFieldEncoder)
    assert isinstance(model.features, FiniteFieldCharacterFeatures)
    assert isinstance(model.features.table, CharacterProbeTable)
    assert isinstance(model.head, CharacterSumHead)

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18

    x = torch.zeros(2, input_channels, 8, 8)
    # Sample A: white-to-move with a small piece set.
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0   # white king e1
    x[0, 11, 0, 4] = 1.0  # black king e8
    x[0, 3, 7, 0] = 1.0   # white rook a1
    x[0, 10, 0, 0] = 1.0  # black queen a8
    x[0, 1, 7, 1] = 1.0   # white knight b1
    # Sample B: black-to-move, different piece configuration.
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0   # white king g1
    x[1, 11, 0, 6] = 1.0  # black king g8
    x[1, 0, 6, 4] = 1.0   # white pawn e2
    x[1, 6, 1, 3] = 1.0   # black pawn d7

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "character_sum_norm",
        "legendre_mean",
        "zero_frequency",
        "residue_entropy",
        "polynomial_value_mean",
        "character_feature_norm",
        "material_balance",
        "piece_count",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key
    assert torch.all(output["character_sum_norm"] >= 0.0)
    assert torch.all(output["character_feature_norm"] >= 0.0)
    assert torch.all(output["zero_frequency"] >= 0.0)
    assert torch.all(output["zero_frequency"] <= 1.0)

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    head_grad = trainable.head.classifier[1].weight.grad
    assert head_grad is not None and torch.isfinite(head_grad).all()

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "finite_field_character_sum_board_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, FiniteFieldCharacterSumBoardNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i067"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i068_schur_ray_line_algebra_network_is_bespoke_and_conformant():
    folder = Path("ideas/i068_schur_ray_line_algebra_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.schur_ray_line_algebra import (
        BoardConditionedLineModes,
        CoordinateBoardStem,
        SchurRayLineAlgebraNetwork,
    )

    assert isinstance(model, SchurRayLineAlgebraNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.stem, CoordinateBoardStem)
    assert isinstance(model.line_modes, BoardConditionedLineModes)

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18

    x = torch.zeros(2, input_channels, 8, 8)
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[0, 3, 7, 0] = 1.0
    x[0, 10, 0, 0] = 1.0
    x[0, 1, 7, 1] = 1.0
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[1, 0, 6, 4] = 1.0
    x[1, 6, 1, 3] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "schur_logdet",
        "schur_trace",
        "line_correction_norm",
        "mean_abs_correction",
        "data_energy",
        "line_energy",
        "king_zone_energy",
        "slider_line_energy",
        "schur_feature_norm",
        "material_balance",
        "piece_count",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    classifier_grad = trainable.classifier[1].weight.grad
    assert classifier_grad is not None and torch.isfinite(classifier_grad).all()

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "schur_ray_line_algebra_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, SchurRayLineAlgebraNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i068"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i069_bitboard_shift_algebra_network_is_bespoke_and_conformant():
    folder = Path("ideas/i069_bitboard_shift_algebra_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.bitboard_shift_algebra import (
        BitboardShiftAlgebraNetwork,
        BitboardStem,
        CoefficientEmitter,
        PATH_NAMES,
        SHIFT_NAMES,
    )

    assert isinstance(model, BitboardShiftAlgebraNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert isinstance(model.stem, BitboardStem)
    assert isinstance(model.coefficients, CoefficientEmitter)
    assert model.shift_maps.shape == (len(SHIFT_NAMES), 64)
    assert len(model.path_names) == len(PATH_NAMES)

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18

    x = torch.zeros(2, input_channels, 8, 8)
    x[0, 12] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[0, 3, 7, 0] = 1.0
    x[0, 10, 0, 0] = 1.0
    x[0, 1, 7, 1] = 1.0
    x[1, 12] = 0.0
    x[1, 5, 7, 6] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[1, 0, 6, 4] = 1.0
    x[1, 6, 1, 3] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "coefficient_entropy",
        "coefficient_abs_mean",
        "top_path_strength",
        "shift_residual",
        "king_zone_shift_residual",
        "occupied_shift_energy",
        "path_output_energy",
        "head_field_energy",
        "cnn_energy",
        "material_balance",
        "piece_count",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    classifier_grad = trainable.classifier[1].weight.grad
    assert classifier_grad is not None and torch.isfinite(classifier_grad).all()

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "bitboard_shift_algebra_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, BitboardShiftAlgebraNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i069"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i074_puzzle_binary_benchmark_challengers_is_bespoke_and_conformant():
    folder = Path("ideas/i074_puzzle_binary_benchmark_challengers")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.puzzle_binary_benchmark_challengers import (
        NegativeClassDisentangledPuzzleHead,
        VALID_ABLATIONS,
        build_negative_class_disentangled_puzzle_head_from_config,
        build_puzzle_binary_benchmark_challengers_from_config,
    )

    assert isinstance(model, NegativeClassDisentangledPuzzleHead)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "puzzle_binary_benchmark_challengers"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    input_channels = int(config["model"]["input_channels"])
    assert input_channels == 18
    x = torch.zeros(2, input_channels, 8, 8)
    x[0, 0, 6, 4] = 1.0
    x[0, 5, 7, 4] = 1.0
    x[0, 6, 1, 3] = 1.0
    x[0, 11, 0, 4] = 1.0
    x[1, 0, 5, 3] = 1.0
    x[1, 4, 6, 2] = 1.0
    x[1, 5, 7, 6] = 1.0
    x[1, 11, 0, 6] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "evidence_random",
        "evidence_near",
        "evidence_puzzle",
        "aux_3way_logits",
        "negative_margin",
        "random_vs_near_gap",
        "trunk_energy",
        "ablation_random_near_merged",
        "ablation_aux_only_no_logsumexp",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output)
    assert output["logits"].shape == (2,)
    assert output["aux_3way_logits"].shape == (2, 3)
    assert torch.isfinite(output["logits"]).all()
    for key in expected_keys:
        assert torch.isfinite(output[key]).all(), key

    # Inference logit must equal the disentangled formula.
    e_random = output["evidence_random"]
    e_near = output["evidence_near"]
    e_puzzle = output["evidence_puzzle"]
    expected_logit = e_puzzle - torch.logsumexp(torch.stack([e_random, e_near], dim=1), dim=1)
    assert torch.allclose(output["logits"], expected_logit, atol=1e-5)
    expected_margin = e_puzzle - torch.maximum(e_random, e_near)
    assert torch.allclose(output["negative_margin"], expected_margin, atol=1e-5)
    expected_gap = (e_random - e_near).abs()
    assert torch.allclose(output["random_vs_near_gap"], expected_gap, atol=1e-5)

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    trunk_grad = trainable.trunk.layers[0].block[0].weight.grad
    proj_grad = trainable.shared_proj[1].weight.grad
    head_random_grad = trainable.head_random.net[0].weight.grad
    head_near_grad = trainable.head_near.net[0].weight.grad
    head_puzzle_grad = trainable.head_puzzle.net[0].weight.grad
    for grad in (trunk_grad, proj_grad, head_random_grad, head_near_grad, head_puzzle_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # The packet's required ablations must build, run, and stay finite.
    expected_ablations = {
        "none",
        "no_aux_3way",
        "random_near_merged",
        "aux_only_no_logsumexp",
        "shuffle_fine_negative_labels",
    }
    assert expected_ablations.issubset(VALID_ABLATIONS)
    for ablation in expected_ablations:
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = build_negative_class_disentangled_puzzle_head_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "random_near_merged":
            assert torch.allclose(abl_out["evidence_random"], abl_out["evidence_near"], atol=1e-6)
            assert torch.all(abl_out["random_vs_near_gap"] < 1e-5)
        if ablation == "aux_only_no_logsumexp":
            assert torch.allclose(abl_out["logits"], abl_out["evidence_puzzle"], atol=1e-5)
        if ablation == "no_aux_3way":
            assert torch.all(abl_out["aux_3way_logits"] == 0.0)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "puzzle_binary_benchmark_challengers"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, NegativeClassDisentangledPuzzleHead)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # Both builder names point at the same callable.
    assert (
        build_puzzle_binary_benchmark_challengers_from_config
        is build_negative_class_disentangled_puzzle_head_from_config
    )

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i074"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i075_tactical_bisimulation_puzzle_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.tactical_bisimulation_puzzle_network import (
        TacticalBisimulationPuzzleNetwork,
        VALID_ABLATIONS,
        build_tactical_bisimulation_puzzle_network_from_config,
    )

    folder = Path("ideas/i075_tactical_bisimulation_puzzle_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, TacticalBisimulationPuzzleNetwork)
    assert not isinstance(model, ResearchPacketProbe)

    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0  # white king
    x[:, 11, 0, 4] = 1.0  # black king
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "base_logit",
        "prototype_distances",
        "min_prototype_distance",
        "soft_min_prototype_distance",
        "mean_prototype_distance",
        "puzzle_prototype_distance",
        "disproof_prototype_distance",
        "random_prototype_distance",
        "successor_signature_entropy",
        "successor_spread",
        "successor_diameter",
        "transition_norm",
        "bisim_residual",
        "move_proposal_entropy",
        "move_attention_entropy",
        "latent_norm",
        "gamma",
        "fine_label_pair_mining_active",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    proto_count = int(config["model"].get("prototype_count", 24))
    assert output["prototype_distances"].shape == (2, proto_count)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    trunk_grad = trainable.stem.layers[0].block[0].weight.grad
    proto_grad = trainable.prototypes.grad
    transition_grad = trainable.transition.net[0].weight.grad
    proposer_grad = trainable.move_proposer.queries.grad
    for grad in (trunk_grad, proto_grad, transition_grad, proposer_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # The packet's required ablations must build, run, and stay finite.
    expected_ablations = {
        "none",
        "no_bisim_loss",
        "no_successor_signature",
        "no_transition_consistency",
        "euclidean_metric_only",
        "random_move_sampler",
        "no_prototypes",
        "binary_margin_only",
        "fine_label_pair_mining_off",
    }
    assert expected_ablations.issubset(VALID_ABLATIONS)
    for ablation in expected_ablations:
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = build_tactical_bisimulation_puzzle_network_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "no_bisim_loss":
            assert torch.all(abl_out["bisim_residual"] == 0.0)
        if ablation == "no_prototypes":
            assert torch.all(abl_out["prototype_distances"] == 0.0)
        if ablation == "no_successor_signature":
            assert torch.all(abl_out["successor_spread"] == 0.0)
            assert torch.all(abl_out["successor_diameter"] == 0.0)
        if ablation == "binary_margin_only":
            assert torch.allclose(abl_out["logits"], abl_out["base_logit"], atol=1e-5)
        if ablation == "fine_label_pair_mining_off":
            assert torch.all(abl_out["fine_label_pair_mining_active"] == 0.0)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "tactical_bisimulation_puzzle_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, TacticalBisimulationPuzzleNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i075"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i076_krylov_tactical_subspace_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.krylov_tactical_subspace_network import (
        KrylovTacticalSubspaceNetwork,
        VALID_ABLATIONS,
        build_krylov_tactical_subspace_network_from_config,
    )

    folder = Path("ideas/i076_krylov_tactical_subspace_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, KrylovTacticalSubspaceNetwork)
    assert not isinstance(model, ResearchPacketProbe)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # side-to-move = white
    x[:, 5, 7, 4] = 1.0  # white king on e1
    x[:, 11, 0, 4] = 1.0  # black king on e8
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "operator_norm",
        "operator_gate_weights",
        "operator_low_rank_energy",
        "role_growth_curves",
        "role_residual_norms",
        "role_ritz_singular_values",
        "role_basis_king_energy",
        "role_basis_target_energy",
        "cross_role_principal_angles",
        "cross_role_gram_frobenius",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()

    num_roles = len(config["model"]["roles"])
    krylov_steps = int(config["model"]["krylov_steps"])
    assert output["role_growth_curves"].shape == (2, num_roles, krylov_steps)
    assert output["role_ritz_singular_values"].shape == (2, num_roles, krylov_steps)
    assert output["role_residual_norms"].shape == (2, num_roles)
    assert output["operator_gate_weights"].shape[1] == 5  # ray, knight, pawn, king, defense
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    trunk_grad = trainable.stem.layers[0].block[0].weight.grad
    seed_grad = trainable.role_seed_head.weight.grad
    gate_grad = trainable.gate_head[0].weight.grad
    low_rank_grad = trainable.low_rank_left.weight.grad
    head_grad = trainable.head[0].weight.grad
    for grad in (trunk_grad, seed_grad, gate_grad, low_rank_grad, head_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # The packet's required ablations must build, run, and stay finite.
    expected_ablations = {
        "none",
        "one_step_only",
        "no_orthogonalization",
        "fixed_operator_only",
        "random_geometry_operator",
        "no_spectral_readout",
        "no_cross_role_angles",
        "cnn_same_params",
    }
    assert expected_ablations.issubset(VALID_ABLATIONS)
    for ablation in expected_ablations:
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = build_krylov_tactical_subspace_network_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "no_spectral_readout":
            assert torch.all(abl_out["role_ritz_singular_values"] == 0.0)
        if ablation == "no_cross_role_angles":
            assert torch.all(abl_out["cross_role_principal_angles"] == 0.0)
            assert torch.all(abl_out["cross_role_gram_frobenius"] == 0.0)
        if ablation == "fixed_operator_only":
            assert torch.all(abl_out["operator_low_rank_energy"] == 0.0)
            uniform = 1.0 / 5.0
            assert torch.allclose(
                abl_out["operator_gate_weights"],
                torch.full_like(abl_out["operator_gate_weights"], uniform),
                atol=1e-6,
            )
        if ablation == "cnn_same_params":
            assert torch.all(abl_out["ablation_cnn_same_params"] == 1.0)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "krylov_tactical_subspace_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, KrylovTacticalSubspaceNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i076"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i077_adaptive_tactical_resolvent_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.adaptive_tactical_resolvent_network import (
        AdaptiveTacticalResolventNetwork,
        VALID_ABLATIONS,
        build_adaptive_tactical_resolvent_network_from_config,
    )

    folder = Path("ideas/i077_adaptive_tactical_resolvent_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, AdaptiveTacticalResolventNetwork)
    assert not isinstance(model, ResearchPacketProbe)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # side-to-move = white
    x[:, 5, 7, 4] = 1.0  # white king on e1
    x[:, 11, 0, 4] = 1.0  # black king on e8
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "operator_norm",
        "operator_gate_weights",
        "operator_low_rank_energy",
        "attack_to_target",
        "defense_to_target",
        "net_pressure",
        "transfer_ratio",
        "resolvent_sensitivity",
        "king_zone_resolvent_energy",
        "material_target_resolvent_energy",
        "alpha_values",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()

    num_alpha = len(config["model"]["alpha_values"])
    assert output["attack_to_target"].shape == (2, num_alpha)
    assert output["defense_to_target"].shape == (2, num_alpha)
    assert output["net_pressure"].shape == (2, num_alpha)
    assert output["transfer_ratio"].shape == (2, num_alpha)
    assert output["resolvent_sensitivity"].shape == (2, num_alpha)
    assert output["king_zone_resolvent_energy"].shape == (2, num_alpha, 3)
    assert output["material_target_resolvent_energy"].shape == (2, num_alpha, 3)
    assert output["operator_gate_weights"].shape[1] == 5  # ray, knight, pawn, king, defense
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    trunk_grad = trainable.stem.layers[0].block[0].weight.grad
    seed_grad = trainable.role_seed_head.weight.grad
    gate_grad = trainable.gate_head[0].weight.grad
    low_rank_grad = trainable.low_rank_left.weight.grad
    alpha_grad = trainable.alpha_logits.grad
    head_grad = trainable.head[0].weight.grad
    for grad in (trunk_grad, seed_grad, gate_grad, low_rank_grad, alpha_grad, head_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # The packet's required ablations must build, run, and stay finite.
    expected_ablations = {
        "none",
        "no_resolvent_direct_pool",
        "neumann_1_step",
        "single_alpha",
        "fixed_operator_no_gates",
        "no_low_rank_update",
        "random_geometry_operator",
        "attack_only_no_defense",
        "cnn_same_params",
    }
    assert expected_ablations.issubset(VALID_ABLATIONS)
    for ablation in expected_ablations:
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = build_adaptive_tactical_resolvent_network_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "fixed_operator_no_gates":
            assert torch.all(abl_out["operator_low_rank_energy"] == 0.0)
            uniform = 1.0 / 5.0
            assert torch.allclose(
                abl_out["operator_gate_weights"],
                torch.full_like(abl_out["operator_gate_weights"], uniform),
                atol=1e-6,
            )
        if ablation == "no_low_rank_update":
            assert torch.all(abl_out["operator_low_rank_energy"] == 0.0)
        if ablation == "single_alpha":
            assert abl_out["alpha_values"].shape == (1,)
        if ablation == "cnn_same_params":
            assert torch.all(abl_out["ablation_cnn_same_params"] == 1.0)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "adaptive_tactical_resolvent_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, AdaptiveTacticalResolventNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i077"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i078_tactical_controllability_gramian_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.tactical_controllability_gramian_network import (
        TacticalControllabilityGramianNetwork,
        VALID_ABLATIONS,
        build_tactical_controllability_gramian_network_from_config,
    )

    folder = Path("ideas/i078_tactical_controllability_gramian_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, TacticalControllabilityGramianNetwork)
    assert not isinstance(model, ResearchPacketProbe)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # side-to-move = white
    x[:, 5, 7, 4] = 1.0  # white king on e1
    x[:, 11, 0, 4] = 1.0  # black king on e8
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "T_a",
        "T_d",
        "T_net",
        "observability_trace",
        "attacker_hankel_modes",
        "defender_hankel_modes",
        "mode_ratio",
        "subspace_principal_angles",
        "target_diag_attacker",
        "target_diag_defender",
        "operator_norm",
        "operator_gate_weights",
        "operator_low_rank_energy",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()

    readout_modes = int(config["model"]["readout_modes"])
    target_rank = int(config["model"]["target_rank"])
    assert output["T_a"].shape == (2,)
    assert output["T_d"].shape == (2,)
    assert output["T_net"].shape == (2,)
    assert output["observability_trace"].shape == (2,)
    assert output["attacker_hankel_modes"].shape == (2, readout_modes)
    assert output["defender_hankel_modes"].shape == (2, readout_modes)
    assert output["mode_ratio"].shape == (2, readout_modes)
    assert output["subspace_principal_angles"].shape == (2, readout_modes)
    assert output["target_diag_attacker"].shape == (2, target_rank)
    assert output["target_diag_defender"].shape == (2, target_rank)
    assert output["operator_gate_weights"].shape[1] == 5  # ray, knight, pawn, king, defense
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    trunk_grad = trainable.stem.layers[0].block[0].weight.grad
    gate_grad = trainable.gate_head[0].weight.grad
    low_rank_grad = trainable.low_rank_left.weight.grad
    attacker_grad = trainable.attacker_input_head.weight.grad
    defender_grad = trainable.defender_input_head.weight.grad
    target_grad = trainable.target_output_head.weight.grad
    head_grad = trainable.head[0].weight.grad
    for grad in (trunk_grad, gate_grad, low_rank_grad, attacker_grad, defender_grad, target_grad, head_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # The packet's required ablations must build, run, and stay finite.
    expected_ablations = {
        "none",
        "attacker_only",
        "defender_only",
        "no_observability",
        "one_step_gramian",
        "random_target_C",
        "random_geometry_A",
        "fixed_A_no_gates",
        "diag_only_gramian",
        "cnn_same_params",
    }
    assert expected_ablations.issubset(VALID_ABLATIONS)
    for ablation in expected_ablations:
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = build_tactical_controllability_gramian_network_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "fixed_A_no_gates":
            assert torch.all(abl_out["operator_low_rank_energy"] == 0.0)
            uniform = 1.0 / 5.0
            assert torch.allclose(
                abl_out["operator_gate_weights"],
                torch.full_like(abl_out["operator_gate_weights"], uniform),
                atol=1e-6,
            )
        if ablation == "attacker_only":
            assert torch.all(abl_out["T_d"] == 0.0)
            assert torch.all(abl_out["target_diag_defender"] == 0.0)
        if ablation == "defender_only":
            assert torch.all(abl_out["T_a"] == 0.0)
            assert torch.all(abl_out["target_diag_attacker"] == 0.0)
        if ablation == "cnn_same_params":
            assert torch.all(abl_out["ablation_cnn_same_params"] == 1.0)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "tactical_controllability_gramian_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, TacticalControllabilityGramianNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i078"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i079_support_polar_zonotope_certificate_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.support_polar_zonotope import (
        SupportPolarZonotopeClassifier,
        VALID_ABLATIONS,
        build_support_polar_zonotope_certificate_network_from_config,
    )

    folder = Path("ideas/i079_support_polar_zonotope_certificate_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, SupportPolarZonotopeClassifier)
    assert not isinstance(model, ResearchPacketProbe)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # side-to-move = white
    x[:, 5, 7, 4] = 1.0  # white king on e1
    x[:, 11, 0, 4] = 1.0  # black king on e8
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "residual",
        "violations",
        "h_plus",
        "h_minus",
        "width",
        "center_projection",
        "proj",
        "U",
        "beta",
        "winning_direction_index",
        "winning_sign",
        "violation_value",
        "operator_scale",
        "auxiliary_logit",
        "gate_mass",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()

    n_dirs = int(config["model"]["n_dirs"])
    d_zono = int(config["model"]["d_zono"])
    assert output["residual"].shape == (2,)
    assert output["violations"].shape == (2, 2 * n_dirs)
    assert output["h_plus"].shape == (2, n_dirs)
    assert output["h_minus"].shape == (2, n_dirs)
    assert output["width"].shape == (2, n_dirs)
    assert output["center_projection"].shape == (2, n_dirs)
    assert output["proj"].shape == (2, 64, 64, n_dirs)
    assert output["U"].shape == (n_dirs, d_zono)
    assert output["beta"].shape == (n_dirs,)
    assert output["winning_direction_index"].shape == (2,)
    assert output["winning_sign"].shape == (2,)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Pair-mask diagonal must be exactly zero (no self-pair generators).
    assert torch.all(output["proj"][:, torch.arange(64), torch.arange(64), :] == 0.0)

    # Closed-form support function: h_plus[k] - h_minus[k] = 2 * <u_k, c_x>.
    cproj = output["center_projection"]
    diff = output["h_plus"] - output["h_minus"]
    assert torch.allclose(diff, 2.0 * cproj, atol=1e-4)

    # Residual is the elementwise max of all violations.
    assert torch.allclose(output["residual"], output["violations"].amax(dim=-1))

    # Brute-force support-function check on a tiny zonotope:
    # for any unit u, h_Z(u) = <u, c> + sum_e |<u, g_e>| with alpha_e in [-1, 1].
    torch.manual_seed(7)
    g = torch.randn(8, d_zono)
    c = torch.randn(d_zono)
    u = torch.randn(d_zono)
    closed = (u @ c) + g.matmul(u).abs().sum()
    # Brute force over alpha in {-1, +1}^E (vertices of [-1,1]^E reach the support).
    best = torch.tensor(float("-inf"))
    for mask in range(1 << 8):
        alpha = torch.tensor([1.0 if (mask >> i) & 1 else -1.0 for i in range(8)])
        z = c + (alpha.unsqueeze(-1) * g).sum(dim=0)
        best = torch.maximum(best, u @ z)
    assert torch.allclose(closed, best, atol=1e-5)

    # Backward through the bespoke pipeline must be finite.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    trunk_grad = trainable.stem.layers[0].block[0].weight.grad
    token_grad = trainable.token_proj.weight.grad
    gen_grad = trainable.gen[0].weight.grad
    gate_grad = trainable.gate[0].weight.grad
    center_grad = trainable.center[0].weight.grad
    dirs_grad = trainable.raw_dirs.grad
    beta_grad = trainable.raw_beta.grad
    scale_grad = trainable.raw_scale.grad
    for grad in (trunk_grad, token_grad, gen_grad, gate_grad, center_grad, dirs_grad, beta_grad, scale_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # The packet's required ablations must build, run, and stay finite.
    expected_ablations = {
        "none",
        "no_zonotope_width",
        "single_square_generators",
        "random_frozen_directions",
        "shared_beta",
        "one_sided",
        "no_relative_encoding",
        "generic_token_baseline",
        "certificate_sanity_check",
    }
    assert expected_ablations.issubset(VALID_ABLATIONS)
    for ablation in expected_ablations:
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = build_support_polar_zonotope_certificate_network_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "no_zonotope_width":
            assert torch.all(abl_out["width"] == 0.0)
        if ablation == "shared_beta":
            assert torch.allclose(
                abl_out["beta"],
                abl_out["beta"][0].expand_as(abl_out["beta"]),
                atol=1e-6,
            )
        if ablation == "random_frozen_directions":
            assert not abl_model.raw_dirs.requires_grad
        if ablation == "no_relative_encoding":
            assert abl_model.rel is None

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "support_polar_zonotope_certificate_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, SupportPolarZonotopeClassifier)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i079"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i080_loop_frustration_curvature_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.loop_frustration_curvature_network import (
        LoopFrustrationCurvatureClassifier,
        VALID_ABLATIONS,
        build_loop_bank,
        build_loop_frustration_curvature_network_from_config,
    )

    folder = Path("ideas/i080_loop_frustration_curvature_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, LoopFrustrationCurvatureClassifier)
    assert not isinstance(model, ResearchPacketProbe)

    # The static graph must match the packet's section-8 counts exactly:
    # M = 210 edges, L = 520 loops, Lmax = Vmax = 12.
    bank = build_loop_bank()
    assert bank["edge_i"].shape == (210,)
    assert bank["edge_j"].shape == (210,)
    assert bank["edge_type"].shape == (210,)
    assert bank["loop_edge_ids"].shape == (520, 12)
    assert bank["loop_edge_mask"].shape == (520, 12)
    assert bank["loop_vertex_ids"].shape == (520, 12)
    assert bank["loop_vertex_mask"].shape == (520, 12)
    edge_type_counts = torch.bincount(bank["edge_type"], minlength=4).tolist()
    assert edge_type_counts == [56, 56, 49, 49]
    loop_lengths = bank["loop_edge_mask"].sum(dim=-1)
    assert int((loop_lengths == 3).sum()) == 196  # triangles
    assert int((loop_lengths == 12).sum()) == 25  # 3x3 rectangles
    # Edges must reference valid sites with matching adjacency.
    for idx in range(bank["edge_i"].shape[0]):
        a = int(bank["edge_i"][idx])
        b = int(bank["edge_j"][idx])
        assert a < b
        ra, ca = divmod(a, 8)
        rb, cb = divmod(b, 8)
        dr = abs(ra - rb)
        dc = abs(ca - cb)
        assert max(dr, dc) == 1

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # side-to-move = white
    x[:, 5, 7, 4] = 1.0  # white king on e1
    x[:, 11, 0, 4] = 1.0  # black king on e8
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "J",
        "loop_product_mid",
        "loop_curvature",
        "loop_omega",
        "omega_site",
        "site_spin",
        "observables",
        "beta",
        "frustration_rate",
        "omega_concentration",
        "ea_order",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()

    replicas = model.replicas
    assert output["J"].shape == (2, replicas, 210)
    assert output["loop_product_mid"].shape == (2, replicas, 520)
    assert output["loop_curvature"].shape == (2, replicas, 520)
    assert output["loop_omega"].shape == (2, replicas, 520)
    assert output["omega_site"].shape == (2, replicas, 8, 8)
    assert output["site_spin"].shape == (2, replicas, 8, 8)
    assert output["observables"].shape == (2, 7 * replicas)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Loop products must stay bounded in [-1, 1] so log(1 + eta * P) is finite.
    p_mid = output["loop_product_mid"]
    assert (p_mid >= -1.0).all()
    assert (p_mid <= 1.0).all()
    # Tanh-bounded couplings must respect the 2.5 clamp.
    assert output["J"].abs().max().item() <= 2.5 + 1e-5

    # Backward through the bespoke pipeline must produce finite gradients.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    encoder_grad = trainable.encoder[0].weight.grad
    spin_grad = trainable.site_spin.weight.grad
    edge_grad = trainable.edge_mlp[0].weight.grad
    beta_grad = trainable.raw_beta.grad
    head_grad = trainable.head[0].weight.grad
    for grad in (encoder_grad, spin_grad, edge_grad, beta_grad, head_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # The packet's required ablations must build, run, and stay finite.
    expected_ablations = {
        "none",
        "no_loop_product",
        "cycle_scramble",
        "no_curvature",
        "no_frustration_weighting",
        "fixed_beta",
        "single_replica",
        "rectangles_only",
        "triangles_only",
    }
    assert expected_ablations.issubset(VALID_ABLATIONS)
    for ablation in expected_ablations:
        abl_cfg = dict(config["model"])
        abl_cfg["ablation"] = ablation
        abl_cfg.pop("name", None)
        abl_model = build_loop_frustration_curvature_network_from_config(abl_cfg).eval()
        with torch.no_grad():
            abl_out = abl_model(x)
        assert abl_out["logits"].shape == (2,), ablation
        assert torch.isfinite(abl_out["logits"]).all(), ablation
        assert abl_model.ablation == ablation
        if ablation == "fixed_beta":
            assert torch.allclose(abl_out["beta"], torch.ones_like(abl_out["beta"]))
        if ablation == "single_replica":
            assert abl_model.replicas == 1
            assert abl_out["loop_omega"].shape == (2, 1, 520)
        if ablation == "rectangles_only":
            assert abl_model.loop_edge_ids.shape[0] == 324
        if ablation == "triangles_only":
            assert abl_model.loop_edge_ids.shape[0] == 196
        if ablation == "no_loop_product":
            # Open-chain magnitudes are non-negative so the surrogate P is in [0, 1].
            assert (abl_out["loop_product_mid"] >= 0.0).all()

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "loop_frustration_curvature_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, LoopFrustrationCurvatureClassifier)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i080"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i081_forcing_response_front_door_bottleneck_is_bespoke_and_conformant():
    from chess_nn_playground.models.forcing_response_front_door_bottleneck import (
        ForcingResponseFrontDoorBottleneck,
        build_forcing_response_front_door_bottleneck_from_config,
    )

    folder = Path("ideas/i081_forcing_response_front_door_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, ForcingResponseFrontDoorBottleneck)
    assert not isinstance(model, ResearchPacketProbe)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # side-to-move = white
    x[:, 5, 7, 4] = 1.0  # white king on e1
    x[:, 11, 0, 4] = 1.0  # black king on e8
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    x[:, 0, 6, 4] = 1.0  # white pawn on e2
    x[:, 6, 1, 4] = 1.0  # black pawn on e7
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "z_c",
        "witness_gates",
        "witness_gate_logits",
        "fine_logits",
        "masked_pred",
        "mechanism_energy",
        "proposal_profile_strength",
        "reply_pressure",
        "defense_gap",
        "sparse_witness_count",
        "sparse_gate_mass",
        "gate_entropy",
        "front_door_bottleneck_l2",
        "top_witness_gate",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    bottleneck_dim = model.to_z.out_features
    max_moves = model.max_moves
    assert output["z_c"].shape == (2, bottleneck_dim)
    assert output["witness_gates"].shape == (2, max_moves)
    assert output["fine_logits"].shape == (2, 3)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # The sparse witness gate must respect the witness budget at eval time.
    active = (output["witness_gates"] > 0).sum(dim=1)
    assert int(active.max().item()) <= model.gate.witness_count

    # The binary head must read only the bottleneck Z_c.
    assert isinstance(model.binary_head[0], torch.nn.Linear)
    assert model.binary_head[0].in_features == bottleneck_dim

    # Backward through the bespoke pipeline must produce finite gradients.
    trainable = module.build_model_from_config(config)
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    stem_grad = trainable.board_stem.input.weight.grad
    move_mlp_grad = trainable.move_mlp.net[0].weight.grad
    response_mlp_grad = trainable.response_mlp.net[0].weight.grad
    gate_grad = trainable.gate.score[1].weight.grad
    head_grad = trainable.binary_head[0].weight.grad
    for grad in (stem_grad, move_mlp_grad, response_mlp_grad, gate_grad, head_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "forcing_response_front_door_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, ForcingResponseFrontDoorBottleneck)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i081"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i082_chess_hypercut_polynomial_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.chess_hypercut_polynomial import (
        ChessHypercutPolynomialNet,
        build_chess_hypercut_polynomial_network_from_config,
    )

    folder = Path("ideas/i082_chess_hypercut_polynomial_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, ChessHypercutPolynomialNet)
    assert not isinstance(model, ResearchPacketProbe)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # white to move
    x[:, 5, 7, 4] = 1.0  # white king on e1
    x[:, 11, 0, 4] = 1.0  # black king on e8
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    x[:, 0, 6, 4] = 1.0  # white pawn on e2
    x[:, 6, 1, 4] = 1.0  # black pawn on e7

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "hyperedge_count",
        "hyperedge_size_mean",
        "hypercut_energy",
        "hypercut_mean",
        "hypercut_max",
        "hypercut_std",
        "higher_order_residual_energy",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Active hyperedge counts must be positive for a board with sliders / kings.
    assert int(output["hyperedge_count"].min().item()) > 0
    # Each active hyperedge must respect the configured size bound.
    assert float(output["hyperedge_size_mean"].max().item()) <= float(model.edge_builder.max_edge_size)

    # Backward through the bespoke pipeline must produce finite gradients.
    trainable = build_chess_hypercut_polynomial_network_from_config(dict(config["model"]))
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    stem_grad = trainable.stem[0].weight.grad
    block_out_grad = trainable.blocks[0].out_weight.grad
    block_probe_grad = trainable.blocks[0].probe.weight.grad
    head_grad = trainable.head[1].weight.grad
    for grad in (stem_grad, block_out_grad, block_probe_grad, head_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "chess_hypercut_polynomial_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, ChessHypercutPolynomialNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i082"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i083_fisher_geodesic_tension_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.fisher_geodesic_tension import (
        FisherGeodesicTensionNet,
        build_fisher_geodesic_tension_network_from_config,
        fisher_geodesic_excess,
        fisher_rao_distance,
    )

    folder = Path("ideas/i083_fisher_geodesic_tension_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, FisherGeodesicTensionNet)
    assert not isinstance(model, ResearchPacketProbe)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # white to move
    x[:, 5, 7, 4] = 1.0  # white king on e1
    x[:, 11, 0, 4] = 1.0  # black king on e8
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    x[:, 0, 6, 4] = 1.0  # white pawn on e2
    x[:, 6, 1, 4] = 1.0  # black pawn on e7

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "geometry_only_logits",
        "route_probs",
        "route_excess",
        "direct_distance",
        "route_ratio",
        "route_gate",
        "weighted_excess",
        "max_excess",
        "weighted_ratio",
        "max_ratio",
        "hinge_turn",
        "weighted_turn",
        "max_turn",
        "geometry_features",
        "fisher_geodesic_tension",
        "information_surprisal",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()

    routes = int(config["model"].get("routes", 8))
    assert output["route_probs"].shape == (2, routes, 3, 64)
    assert output["route_excess"].shape == (2, routes)
    assert output["route_gate"].shape == (2, routes)

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Route distributions must lie on the categorical simplex.
    probs = output["route_probs"]
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    assert (probs > 0).all()

    # Fisher-Rao geodesic excess is nonnegative within numerical tolerance.
    assert (output["route_excess"] >= -1e-4).all()
    # Direct distance is always at least as long as the excess differential
    # implies, i.e. the path-distance bound holds.
    assert (
        output["route_excess"]
        <= output["direct_distance"] + 2.0 * float(torch.pi) + 1e-4
    ).all()

    # Helper functions agree on a hand-built example.
    p = torch.tensor([[0.7, 0.2, 0.1]])
    q = torch.tensor([[0.7, 0.2, 0.1]])
    # Bhattacharyya coefficient is clamped to 1 - eps, so d_FR(p, p) is a
    # small positive number rather than exactly zero.
    assert (fisher_rao_distance(p, q) < 1e-2).all()
    p = torch.tensor([[0.5, 0.3, 0.2]])
    h = torch.tensor([[0.4, 0.4, 0.2]])
    q = torch.tensor([[0.3, 0.5, 0.2]])
    excess, _, _, _ = fisher_geodesic_excess(p, h, q)
    assert (excess >= -1e-5).all()
    # Symmetry of Fisher-Rao distance.
    assert torch.allclose(
        fisher_rao_distance(p, q), fisher_rao_distance(q, p), atol=1e-6
    )

    # Backward through the bespoke pipeline must produce finite gradients
    # for the convolutional trunk, route head, route gate, readout, and the
    # geometry-only ablation head.
    trainable = build_fisher_geodesic_tension_network_from_config(dict(config["model"]))
    trainable_out = trainable(x)
    (trainable_out["logits"].sum() + trainable_out["geometry_only_logits"].sum()).backward()
    stem_grad = trainable.stem[0].weight.grad
    route_head_grad = trainable.route_head.weight.grad
    route_gate_grad = trainable.route_gate[1].weight.grad
    readout_grad = trainable.readout[1].weight.grad
    geom_only_grad = trainable.geometry_only_head[1].weight.grad
    for grad in (stem_grad, route_head_grad, route_gate_grad, readout_grad, geom_only_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "fisher_geodesic_tension_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, FisherGeodesicTensionNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i083"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i084_typed_hypergraph_motif_grammar_is_bespoke_and_conformant():
    from chess_nn_playground.models.typed_hypergraph_motif_grammar import (
        TypedHypergraphMotifGrammarNet,
        build_typed_hypergraph_motif_grammar_from_config,
    )

    folder = Path("ideas/i084_typed_hypergraph_motif_grammar")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, TypedHypergraphMotifGrammarNet)
    assert not isinstance(model, ResearchPacketProbe)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # white to move
    x[:, 5, 7, 4] = 1.0  # white king on e1
    x[:, 11, 0, 4] = 1.0  # black king on e8
    x[:, 3, 7, 0] = 1.0  # white rook on a1
    x[:, 10, 0, 0] = 1.0  # black queen on a8
    x[:, 0, 6, 4] = 1.0  # white pawn on e2
    x[:, 6, 1, 4] = 1.0  # black pawn on e7

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "grammar_only_logits",
        "motif_summary",
        "pressure_motif_strength",
        "loose_target_strength",
        "king_zone_pressure_strength",
        "pin_shape_strength",
        "line_pressure_strength",
        "fork_shape_strength",
        "battery_shape_strength",
        "compromised_defender_strength",
        "overload_shape_strength",
        "tactical_convergence_strength",
        "puzzle_like_motif_strength",
        "grammar_chart_energy",
        "motif_entropy",
        "relation_fact_count",
        "piece_count",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "grammar_composition_depth",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert output["grammar_only_logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # The deterministic relation extractor should see exactly the six
    # pieces placed on the board for both batch entries.
    assert torch.equal(output["piece_count"], torch.tensor([6.0, 6.0]))
    # Per-production strength scalars are bounded sigmoids.
    for name in (
        "pressure_motif_strength",
        "loose_target_strength",
        "king_zone_pressure_strength",
        "pin_shape_strength",
        "line_pressure_strength",
        "fork_shape_strength",
        "battery_shape_strength",
        "compromised_defender_strength",
        "overload_shape_strength",
        "tactical_convergence_strength",
        "puzzle_like_motif_strength",
    ):
        assert output[name].shape == (2,)
        assert ((output[name] >= 0.0) & (output[name] <= 1.0)).all(), name

    # The auxiliary production-mass tensor is exposed when requested.
    aux_output = model(x, return_aux=True)
    assert aux_output["production_mass"].shape == (2, 11)

    # Backward through the bespoke pipeline must produce finite gradients
    # for the convolutional trunk, piece encoder, pair scorer, grammar-only
    # head, and the fused readout.
    trainable = build_typed_hypergraph_motif_grammar_from_config(dict(config["model"]))
    trainable_out = trainable(x)
    (trainable_out["logits"].sum() + trainable_out["grammar_only_logits"].sum()).backward()
    stem_grad = trainable.board_stem[0].weight.grad
    piece_encoder_grad = trainable.piece_encoder[1].weight.grad
    pair_scorer_grad = trainable.pair_scorer[1].weight.grad
    grammar_only_grad = trainable.grammar_only_head[0].weight.grad
    head_grad = trainable.head[1].weight.grad
    production_bias_grad = trainable.production_bias.grad
    for grad in (
        stem_grad,
        piece_encoder_grad,
        pair_scorer_grad,
        grammar_only_grad,
        head_grad,
        production_bias_grad,
    ):
        assert grad is not None and torch.isfinite(grad).all()

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "typed_hypergraph_motif_grammar"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, TypedHypergraphMotifGrammarNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i084"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i086_differentiable_chess_fact_lattice_is_bespoke_and_conformant():
    from chess_nn_playground.models.differentiable_chess_fact_lattice import (
        DifferentiableChessFactLatticeNet,
        DifferentiableFactInterpreter,
        build_differentiable_chess_fact_lattice_from_config,
    )

    folder = Path("ideas/i086_differentiable_chess_fact_lattice")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, DifferentiableChessFactLatticeNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "differentiable_chess_fact_lattice"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — produces a non-trivial pin/overload pattern on a1.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key in (
        "abstract_features",
        "abstract_feature_energy",
        "interval_width_mean",
        "widening_width",
        "conflict_energy",
        "attack_mass",
        "defense_mass",
        "king_zone_pressure",
        "value_at_risk",
        "line_exposure",
        "board_consistency_error",
        "monotonicity_penalty",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    ):
        assert key in output, key
        assert torch.isfinite(output[key]).all(), key
    interpreter = model.interpreter
    expected_channels = interpreter.abstract_feature_channels
    assert output["abstract_features"].shape == (2, expected_channels, 8, 8)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "differentiable_chess_fact_lattice"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, DifferentiableChessFactLatticeNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Backward through the abstract-interpretation bottleneck must yield
    # finite gradients on the convolutional readout, the puzzle head, and
    # the learned per-piece attack gate.
    trainable = build_differentiable_chess_fact_lattice_from_config(dict(config["model"]))
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    readout_grad = trainable.readout[0].weight.grad
    head_grad = trainable.head[0].weight.grad
    gate_grad = trainable.interpreter.piece_attack_gate.grad
    for grad in (readout_grad, head_grad, gate_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # Ablation switches from the math thesis must build and produce finite
    # logits at the puzzle_binary contract shape.
    for ablation in (
        {"use_intervals": False},
        {"use_meet_channels": False},
        {"use_ray_transfer": False},
        {"use_king_zone": False},
        {"variant": "pool_control"},
    ):
        cfg = {**dict(config["model"]), **ablation}
        cfg.pop("name", None)
        ablated = build_differentiable_chess_fact_lattice_from_config(cfg).eval()
        with torch.no_grad():
            ablated_out = ablated(x)
        assert ablated_out["logits"].shape == (2,), ablation
        assert torch.isfinite(ablated_out["logits"]).all(), ablation

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i086"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i087_tactical_radius_filtration_is_bespoke_and_conformant():
    from chess_nn_playground.models.tactical_radius_filtration import (
        TacticalRadiusFiltrationClassifier,
        TacticalRadiusGraphBuilder,
        build_tactical_radius_filtration_from_config,
    )

    folder = Path("ideas/i087_tactical_radius_filtration")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, TacticalRadiusFiltrationClassifier)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "tactical_radius_filtration"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — produces a non-trivial pin/blocker pattern around a1.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    for key in (
        "radius_shell_counts",
        "shell_readout_features",
        "piece_pool_energy",
        "topology_pressure",
        "radius2_pressure",
        "radius3_pressure",
        "shell_count_hint",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    ):
        assert key in output, key
        assert torch.isfinite(output[key]).all(), key
    assert output["radius_shell_counts"].shape == (2, model.radius + 1)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "tactical_radius_filtration"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, TacticalRadiusFiltrationClassifier)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Backward through the shell filtration must yield finite gradients on
    # the per-square lift, the typed shell projections, and the readout head.
    trainable = build_tactical_radius_filtration_from_config(dict(config["model"]))
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    lift_grad = trainable.square_lift[0].weight.grad
    self_proj_grad = trainable.self_projections[1].weight.grad
    group_proj_grad = trainable.group_projections[1][0].weight.grad
    readout_grad = trainable.readout[1].weight.grad
    for grad in (lift_grad, self_proj_grad, group_proj_grad, readout_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # Math-thesis ablation switches must build and return finite logits at
    # the puzzle_binary contract shape.
    for ablation in (
        {"radius": 1},
        {"shell_mode": "closed_ball"},
        {"graph_mode": "chebyshev"},
        {"use_xray": False},
        {"use_king_zone": False},
        {"use_shell_counts": False},
    ):
        cfg = {**dict(config["model"]), **ablation}
        cfg.pop("name", None)
        ablated = build_tactical_radius_filtration_from_config(cfg).eval()
        with torch.no_grad():
            ablated_out = ablated(x)
        assert ablated_out["logits"].shape == (2,), ablation
        assert torch.isfinite(ablated_out["logits"]).all(), ablation

    # The graph builder is deterministic and produces typed shells per radius.
    builder = TacticalRadiusGraphBuilder(max_radius=int(config["model"].get("radius", 3)))
    graph = builder.build(x)
    assert len(graph.groups_by_radius) == builder.max_radius + 1
    assert graph.masks.shape == (2, 6, 64)

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i087"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i090_chess_mode_tucker_relation_certificate_is_bespoke_and_conformant():
    folder = Path("ideas/i090_chess_mode_tucker_relation_certificate")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    from chess_nn_playground.models.chess_mode_tucker_relation_certificate import (
        ChessModeTuckerRelationCertificate,
        FlatProjectedMLPControl,
        count_trainable_parameters,
        fine_label_diagnostic_3x2,
    )

    assert isinstance(model, ChessModeTuckerRelationCertificate)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "chess_mode_tucker_relation_certificate"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 0, 6, 4] = 1.0
    x[:, 3, 5, 2] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 6, 1, 4] = 1.0
    x[:, 10, 2, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        relation_tensor = model.relation_tensor(x)
        projected = model.tucker_project(relation_tensor)
        output = model(x)

    assert relation_tensor.shape == (2, model.latent_channels, 12, 8, 10)
    assert projected.shape == (2, *model.rank_shape)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert {
        "logits",
        "rank_certificate",
        "K_mode_eff_rank",
        "R_mode_eff_rank",
        "D_mode_eff_rank",
        "G_mode_eff_rank",
        "nuclear_bottleneck",
        "orthogonality_penalty",
        "fixed_relation_density",
        "region_mass_error",
    }.issubset(output)
    assert output["rank_certificate"].shape == (2, 4)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    counts, rates = fine_label_diagnostic_3x2(
        torch.tensor([2.0, -2.0, 2.0]),
        torch.tensor([0, 1, 2]),
    )
    assert counts.shape == (3, 2)
    assert rates.shape == (3, 2)
    assert int(counts.sum()) == 3

    paper_main = ChessModeTuckerRelationCertificate(input_channels=input_channels)
    paper_control = FlatProjectedMLPControl(input_channels=input_channels)
    assert count_trainable_parameters(paper_control) == count_trainable_parameters(paper_main)
    with torch.no_grad():
        control_output = paper_control(x)
    assert control_output["logits"].shape == (2,)
    assert torch.isfinite(control_output["logits"]).all()

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    registry_model = build_model(model_name, model_cfg).eval()
    assert isinstance(registry_model, ChessModeTuckerRelationCertificate)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i090"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i091_tactical_state_bottleneck_inference_is_bespoke_and_conformant():
    from chess_nn_playground.models.tactical_state_bottleneck import (
        LATENT_SIZES,
        NoLatentMatchedBaseline,
        TacticalStateBottleneckModel,
        build_tactical_state_bottleneck_from_config,
        diagnostic_3x2,
        kl_freebits,
        tactical_state_loss_components,
    )

    folder = Path("ideas/i091_tactical_state_bottleneck_inference")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, TacticalStateBottleneckModel)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "tactical_state_bottleneck_inference"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — a board with a non-trivial tactical context.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "logit",
        "prob",
        "latent_probs",
        "prior_entropy_by_group",
        "prior_usage_by_group",
        "motif_entropy",
        "anchor_entropy",
        "target_entropy",
        "relation_entropy",
        "vulnerability_entropy",
        "tempo_entropy",
        "motif_usage",
        "relation_usage",
        "vulnerability_usage",
        "tempo_usage",
        "anchor_null_rate",
        "target_null_rate",
        "pooled_energy",
        "direct_alpha",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    }
    assert expected_keys.issubset(output)
    for name, size in LATENT_SIZES.items():
        assert output["latent_probs"][name].shape == (2, size), name
    assert output["anchor_null_rate"].shape == (2,)
    assert output["target_null_rate"].shape == (2,)

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "tactical_state_bottleneck_inference"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, TacticalStateBottleneckModel)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # The prior path must produce finite gradients on the trunk and the
    # latent-conditioned puzzle head.
    trainable = build_tactical_state_bottleneck_from_config(dict(config["model"]))
    trainable.train()
    output = trainable(x)
    output["logits"].sum().backward()
    trunk_grad = trainable.trunk.input.weight.grad
    motif_grad = trainable.prior_head.motif.weight.grad
    motif_emb_grad = trainable.latents["motif"].embedding.grad
    head_grad = trainable.latent_head[1].weight.grad
    direct_grad = trainable.direct_head[1].weight.grad
    for grad in (trunk_grad, motif_grad, motif_emb_grad, head_grad, direct_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # The full posterior-aware path must build, return finite logits at the
    # puzzle_binary contract shape, and produce a finite multi-loss bundle.
    trainable.train()
    fine_label = torch.tensor([2, 0], dtype=torch.long)
    train_output = trainable(x, fine_label=fine_label)
    assert train_output["logits"].shape == (2,)
    for key in ("logit_q", "logit_p", "prior_logits", "posterior_logits", "losses"):
        assert key in train_output, key
    losses = train_output["losses"]
    for key in ("loss", "loss_pred", "loss_prior_pred", "loss_kl", "loss_usage", "loss_entropy"):
        assert key in losses, key
        assert torch.isfinite(losses[key]).all(), key
    standalone = tactical_state_loss_components(
        logit_q=train_output["logit_q"],
        logit_p=train_output["logit_p"],
        posterior_logits=train_output["posterior_logits"],
        prior_logits=train_output["prior_logits"],
        fine_label=fine_label,
    )
    assert torch.isfinite(standalone["loss"]).all()
    kl_total, kl_by_group = kl_freebits(train_output["posterior_logits"], train_output["prior_logits"])
    assert torch.isfinite(kl_total).all()
    assert set(kl_by_group) == set(LATENT_SIZES)

    # The 3x2 fine-label diagnostic must be a [3, 2] long tensor with the right
    # per-row totals.
    diag_table = diagnostic_3x2(torch.tensor([0, 1, 2, 2]), torch.tensor([2.0, -2.0, 2.0, -2.0]))
    assert diag_table.shape == (3, 2)
    assert int(diag_table.sum()) == 4

    # The matched no-latent baseline must build and return the puzzle_binary
    # contract shape so ablations can run beside the main model.
    baseline = NoLatentMatchedBaseline(input_channels=input_channels)
    with torch.no_grad():
        baseline_output = baseline(x)
    assert baseline_output["logits"].shape == (2,)
    assert torch.isfinite(baseline_output["logits"]).all()

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i091"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i088_traced_threat_motif_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.traced_threat_motif import (
        MOTIF_WORDS,
        TracedThreatMotifNet,
        build_traced_threat_motif_network_from_config,
    )

    folder = Path("ideas/i088_traced_threat_motif_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, TracedThreatMotifNet)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "traced_threat_motif_network"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — produces a non-trivial pin/fork pattern around a1.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "motif_scores",
        "top_motif_idx",
        "contest_heatmap",
        "contest_features",
        "trace_closure",
        "open_king_mass",
        "open_value_mass",
        "monoidal_features",
        "parallel_loop2",
        "interaction_loop",
        "raw_relation_density",
        "gated_relation_density",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    }
    assert expected_keys.issubset(output)
    assert output["motif_scores"].shape == (2, len(MOTIF_WORDS))
    assert output["contest_heatmap"].shape == (2, 8, 8)
    assert output["contest_features"].shape == (2, 3)
    assert output["monoidal_features"].shape == (2, 4)

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "traced_threat_motif_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, TracedThreatMotifNet)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Backward through the motif algebra must yield finite gradients on the
    # trunk, the relation gate, the group-mixing softmax, and the head.
    trainable = build_traced_threat_motif_network_from_config(dict(config["model"]))
    trainable.train()
    trainable_out = trainable(x)
    trainable_out["logits"].sum().backward()
    stem_grad = trainable.stem.input.weight.grad
    gate_q_grad = trainable.relation_gate.wq.grad
    mixer_grad = trainable.group_mixer.group_logits.grad
    value_grad = trainable.motif_composer.value_weight.grad
    head_grad = trainable.head[1].weight.grad
    for grad in (stem_grad, gate_q_grad, mixer_grad, value_grad, head_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # The gated relations are row-substochastic on the geometry mask.
    with torch.no_grad():
        diag = trainable.eval()(x, return_diag=True)
    assert "group_ctrl_mass" in diag
    assert diag["group_ctrl_mass"].shape == (2, 2)

    # Width / depth ablation switches must build and produce finite logits at
    # the puzzle_binary contract shape.
    for ablation in (
        {"d_model": 64},
        {"stem_blocks": 1},
        {"head_hidden": 128},
        {"gate_dim": 16},
        {"dropout": 0.0},
    ):
        cfg = {**dict(config["model"]), **ablation}
        cfg.pop("name", None)
        ablated = build_traced_threat_motif_network_from_config(cfg).eval()
        with torch.no_grad():
            ablated_out = ablated(x)
        assert ablated_out["logits"].shape == (2,), ablation
        assert torch.isfinite(ablated_out["logits"]).all(), ablation

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i088"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i092_parity_syndrome_puzzle_bottleneck_is_bespoke_and_conformant():
    from chess_nn_playground.models.parity_syndrome import (
        LiteralEncoder,
        ParityCheckBank,
        ParitySyndromePuzzleBottleneck,
        SyndromeStats,
        build_parity_syndrome_puzzle_bottleneck_from_config,
    )

    folder = Path("ideas/i092_parity_syndrome_puzzle_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, ParitySyndromePuzzleBottleneck)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "parity_syndrome_puzzle_bottleneck"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES
    assert isinstance(model.encoder, LiteralEncoder)
    assert isinstance(model.check_bank, ParityCheckBank)
    assert isinstance(model.stats, SyndromeStats)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — produces a non-trivial pin/fork pattern.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "syndromes",
        "syndrome_features",
        "literal_mean",
        "literal_entropy",
        "parity_check_mode",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "syndrome_mean",
        "syndrome_std",
        "syndrome_max",
        "syndrome_margin_mean",
        "syndrome_margin_max",
        "syndrome_entropy",
        "check_degree_mean",
        "check_gate_density",
        "top_syndrome_values",
        "top_syndrome_margins",
        "syndrome_histogram",
        "margin_histogram",
    }
    assert expected_keys.issubset(output)
    assert output["syndromes"].shape == (2, model.check_bank.num_checks)
    assert (output["syndromes"] >= 0).all() and (output["syndromes"] <= 1).all()
    with torch.no_grad():
        diag_output = model(x, return_diag=True)
    assert "literal_probs" in diag_output and "check_gates" in diag_output
    assert diag_output["check_gates"].shape == (
        model.check_bank.num_checks,
        model.num_literals,
    )

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "parity_syndrome_puzzle_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, ParitySyndromePuzzleBottleneck)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Gradients must flow into the literal encoder, the parity gates, and the head.
    trainable = build_parity_syndrome_puzzle_bottleneck_from_config(dict(config["model"]))
    trainable.train()
    train_output = trainable(x)
    train_output["logits"].sum().backward()
    encoder_grad = trainable.encoder.net[0].weight.grad
    left_grad = trainable.check_bank.left.grad
    right_grad = trainable.check_bank.right.grad
    head_grad = trainable.head[1].weight.grad
    for grad in (encoder_grad, left_grad, right_grad, head_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # Ablation modes must build and produce the puzzle_binary contract shape.
    for mode in ("sum_checks", "random_parity_checks", "dense_parity_no_sparsity"):
        ablation_cfg = dict(config["model"])
        ablation_cfg.pop("name", None)
        ablation_cfg.pop("packet_profile", None)
        ablation_cfg.pop("mechanism_family", None)
        ablation_cfg["mode"] = mode
        ablation_model = build_parity_syndrome_puzzle_bottleneck_from_config(ablation_cfg).eval()
        with torch.no_grad():
            ablation_output = ablation_model(x)
        assert ablation_output["logits"].shape == (2,), mode
        assert torch.isfinite(ablation_output["logits"]).all(), mode

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i092"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i093_wavelet_scattering_board_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.wavelet_scattering_board_network import (
        FixedWaveletBank,
        WaveletScatteringBoardNetwork,
        WaveletScatteringFeatures,
        build_wavelet_scattering_board_network_from_config,
    )

    folder = Path("ideas/i093_wavelet_scattering_board_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, WaveletScatteringBoardNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "wavelet_scattering_board_network"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES
    assert isinstance(model.scattering, WaveletScatteringFeatures)
    assert all(isinstance(bank, FixedWaveletBank) for bank in model.scattering.first_layer)

    # Filter buffers must not be learnable.
    parameter_ids = {id(p) for p in model.parameters()}
    for bank in model.scattering.first_layer:
        assert id(bank.weight) not in parameter_ids

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # Sparse but distinguishable test position.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "scattering_features",
        "first_order_mean_field",
        "first_order_std_field",
        "first_order_max_field",
        "lowpass_energy",
        "second_order_mean_field",
        "scattering_mode",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
        "scale_count",
    }
    assert expected_keys.issubset(output)
    assert output["scattering_features"].shape == (2, model.feature_dim)
    assert output["first_order_mean_field"].shape == (
        2,
        input_channels,
        model.scattering.num_scales,
        model.scattering.num_orientations,
    )
    assert output["lowpass_energy"].shape == (2, input_channels, model.scattering.num_scales)

    with torch.no_grad():
        diag_output = model(x, return_diag=True)
    assert "first_order_modulus" in diag_output and "lowpass_field" in diag_output
    assert diag_output["first_order_modulus"].shape == (
        2,
        input_channels,
        model.scattering.num_scales,
        model.scattering.num_orientations,
        8,
        8,
    )

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "wavelet_scattering_board_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, WaveletScatteringBoardNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Gradients must flow into the head; wavelet filters must remain frozen.
    trainable = build_wavelet_scattering_board_network_from_config(dict(config["model"]))
    trainable.train()
    train_output = trainable(x)
    train_output["logits"].sum().backward()
    head_grads = [p.grad for p in trainable.head.parameters() if p.requires_grad]
    assert head_grads and all(g is not None and torch.isfinite(g).all() for g in head_grads)
    for bank in trainable.scattering.first_layer:
        assert not bank.weight.requires_grad

    # Ablation modes must build and produce the puzzle_binary contract shape.
    for mode in ("random_fixed_filters", "lowpass_only", "channel_shuffle"):
        ablation_cfg = dict(config["model"])
        ablation_cfg.pop("name", None)
        ablation_cfg.pop("packet_profile", None)
        ablation_cfg.pop("mechanism_family", None)
        ablation_cfg["mode"] = mode
        ablation_model = build_wavelet_scattering_board_network_from_config(ablation_cfg).eval()
        with torch.no_grad():
            ablation_output = ablation_model(x)
        assert ablation_output["logits"].shape == (2,), mode
        assert torch.isfinite(ablation_output["logits"]).all(), mode
        assert ablation_output["scattering_mode"][0].item() == float(
            WaveletScatteringBoardNetwork.MODES.index(mode)
        )

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i093"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i094_convex_feasibility_residual_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.convex_feasibility import (
        BoardFeasibilityEncoder,
        ConvexFeasibilityResidualNetwork,
        LearnedConvexConstraints,
        MaterialOnlyEncoder,
        SoftProjectionLayer,
        build_convex_feasibility_residual_network_from_config,
    )

    folder = Path("ideas/i094_convex_feasibility_residual_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, ConvexFeasibilityResidualNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "convex_feasibility_residual_network"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES
    assert isinstance(model.encoder, BoardFeasibilityEncoder)
    assert isinstance(model.constraints, LearnedConvexConstraints)
    assert isinstance(model.projector, SoftProjectionLayer)
    assert isinstance(model.material_encoder, MaterialOnlyEncoder)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — produces a non-trivial pin/fork pattern.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "z",
        "projected_z",
        "feasibility_residual",
        "violations",
        "halfspace_violations",
        "ball_violations",
        "path_step_norms",
        "path_length",
        "residual_norm",
        "max_violation",
        "mean_violation",
        "feasibility_energy",
        "feasible_fraction",
        "halfspace_feasible_fraction",
        "ball_feasible_fraction",
        "constraint_gate_mean",
        "projection_mode",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    }
    assert expected_keys.issubset(output)
    num_constraints = model.constraints.num_constraints
    assert output["violations"].shape == (2, num_constraints)
    assert (output["violations"] >= 0).all()
    assert output["projected_z"].shape == (2, model.latent_dim)
    assert output["path_step_norms"].shape == (2, model.projector.projection_steps)

    with torch.no_grad():
        diag_output = model(x, return_projection=True)
    assert "projection_path" in diag_output and "constraint_gates" in diag_output
    assert diag_output["projection_path"].shape == (
        2,
        model.projector.projection_steps,
        model.latent_dim,
    )

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "convex_feasibility_residual_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, ConvexFeasibilityResidualNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Gradients must flow into the encoder, the constraint bank, and the head.
    trainable_cfg = dict(config["model"])
    trainable_cfg.pop("name", None)
    trainable_cfg.pop("packet_profile", None)
    trainable_cfg.pop("mechanism_family", None)
    trainable = build_convex_feasibility_residual_network_from_config(trainable_cfg)
    trainable.train()
    train_output = trainable(x)
    train_output["logits"].sum().backward()
    encoder_conv = next(
        m for m in trainable.encoder.conv if isinstance(m, torch.nn.Conv2d)
    )
    constraint_grad = trainable.constraints.halfspace_normals.grad
    head_grad = trainable.residual_head[1].weight.grad
    for grad in (encoder_conv.weight.grad, constraint_grad, head_grad):
        assert grad is not None and torch.isfinite(grad).all()

    # Ablation modes must build and produce the puzzle_binary contract shape.
    for mode in ("no_projection", "random_constraints", "linear_head_same_params", "material_only_encoder"):
        ablation_cfg = dict(config["model"])
        ablation_cfg.pop("name", None)
        ablation_cfg.pop("packet_profile", None)
        ablation_cfg.pop("mechanism_family", None)
        ablation_cfg["mode"] = mode
        ablation_model = build_convex_feasibility_residual_network_from_config(ablation_cfg).eval()
        with torch.no_grad():
            ablation_output = ablation_model(x)
        assert ablation_output["logits"].shape == (2,), mode
        assert torch.isfinite(ablation_output["logits"]).all(), mode

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i094"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i095_rank_quantile_evidence_field_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.rank_quantile import (
        EvidenceFieldEncoder,
        RankQuantileEvidenceFieldNetwork,
        RankQuantilePooler,
        build_rank_quantile_evidence_field_network_from_config,
    )

    folder = Path("ideas/i095_rank_quantile_evidence_field_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, RankQuantileEvidenceFieldNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "rank_quantile_evidence_field_network"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES
    assert isinstance(model.encoder, EvidenceFieldEncoder)
    assert isinstance(model.pooler, RankQuantilePooler)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — a non-trivial pin/fork pattern.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "rank_features",
        "material_safe_stats",
        "quantiles",
        "tail_gaps",
        "field_mean",
        "field_std",
        "topk_means",
        "bottomk_means",
        "rank_entropy",
        "robust_range",
        "high_tail_mass",
        "low_tail_mass",
        "extreme_gap_mean",
        "upper_tail_gap",
        "lower_tail_gap",
        "median_evidence",
        "max_quantile_evidence",
        "min_quantile_evidence",
        "field_energy",
        "rank_readout_mode",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    }
    assert expected_keys.issubset(output)
    evidence_fields = model.pooler.evidence_fields
    quantile_count = model.pooler.quantile_count
    assert output["quantiles"].shape == (2, evidence_fields, quantile_count)
    assert output["tail_gaps"].shape == (2, evidence_fields, 4)
    assert output["rank_features"].shape == (2, model.pooler.output_dim)

    with torch.no_grad():
        diag_output = model(x, return_fields=True)
    assert "evidence_fields" in diag_output and "readout_features" in diag_output
    assert diag_output["evidence_fields"].shape == (2, evidence_fields, 8, 8)

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "rank_quantile_evidence_field_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, RankQuantileEvidenceFieldNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Gradients must flow into the encoder and the head.
    trainable_cfg = dict(config["model"])
    trainable_cfg.pop("name", None)
    trainable_cfg.pop("packet_profile", None)
    trainable_cfg.pop("mechanism_family", None)
    trainable = build_rank_quantile_evidence_field_network_from_config(trainable_cfg)
    trainable.train()
    train_output = trainable(x)
    train_output["logits"].sum().backward()
    encoder_conv = next(
        m for m in trainable.encoder.learned if isinstance(m, torch.nn.Conv2d)
    )
    head_linear = next(
        m for m in trainable.head if isinstance(m, torch.nn.Linear)
    )
    for grad in (encoder_conv.weight.grad, head_linear.weight.grad):
        assert grad is not None and torch.isfinite(grad).all()

    # Ablation modes must build and produce the puzzle_binary contract shape.
    for mode in ("mean_pool_only", "topk_only", "random_field_encoder", "square_shuffle"):
        ablation_cfg = dict(config["model"])
        ablation_cfg.pop("name", None)
        ablation_cfg.pop("packet_profile", None)
        ablation_cfg.pop("mechanism_family", None)
        ablation_cfg["mode"] = mode
        ablation_model = build_rank_quantile_evidence_field_network_from_config(ablation_cfg).eval()
        with torch.no_grad():
            ablation_output = ablation_model(x)
        assert ablation_output["logits"].shape == (2,), mode
        assert torch.isfinite(ablation_output["logits"]).all(), mode

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i095"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i096_oriented_matroid_covector_bottleneck_is_bespoke_and_conformant():
    from chess_nn_playground.models.oriented_matroid_covector import (
        CovectorStats,
        HyperplaneArrangement,
        OccupiedPieceTokenizer,
        OrientedMatroidCovectorBottleneck,
        PieceTokenEncoder,
        build_oriented_matroid_covector_bottleneck_from_config,
    )

    folder = Path("ideas/i096_oriented_matroid_covector_bottleneck")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, OrientedMatroidCovectorBottleneck)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "oriented_matroid_covector_bottleneck"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES
    assert isinstance(model.tokenizer, OccupiedPieceTokenizer)
    assert isinstance(model.token_encoder, PieceTokenEncoder)
    assert isinstance(model.arrangement, HyperplaneArrangement)
    assert isinstance(model.stats, CovectorStats)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — a non-trivial pin/fork pattern.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "covector_features",
        "token_mask",
        "piece_count",
        "soft_signs",
        "hyperplane_scores",
        "positive_counts",
        "negative_counts",
        "near_zero_counts",
        "sign_agreement",
        "role_sign_entropy",
        "role_histogram",
        "sign_mean",
        "sign_abs_mean",
        "score_abs_mean",
        "score_std",
        "covector_entropy",
        "near_zero_rate",
        "pairwise_agreement_energy",
        "orientation_mode",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    }
    assert expected_keys.issubset(output)
    hyperplanes = model.arrangement.hyperplanes
    max_pieces = model.tokenizer.max_pieces
    assert output["hyperplane_scores"].shape == (2, max_pieces, hyperplanes)
    assert output["soft_signs"].shape == (2, max_pieces, hyperplanes)
    assert output["sign_agreement"].shape == (2, hyperplanes, hyperplanes)
    assert output["role_sign_entropy"].shape == (2, 12, hyperplanes)
    assert output["role_histogram"].shape == (2, 12)
    assert output["covector_features"].shape == (2, model.stats.output_dim)
    # Soft sign output must respect the [-1, 1] tanh bounds and the token mask.
    mask = output["token_mask"]
    assert torch.all(output["soft_signs"].abs() <= 1.0 + 1e-5)
    padded = output["soft_signs"][mask == 0]
    assert torch.allclose(padded, torch.zeros_like(padded))

    with torch.no_grad():
        diag_output = model(x, return_covectors=True)
    assert {"token_features", "token_embeddings", "role_probs", "square_indices"}.issubset(diag_output)
    assert diag_output["token_embeddings"].shape == (2, max_pieces, model.arrangement.token_dim)

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "oriented_matroid_covector_bottleneck"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, OrientedMatroidCovectorBottleneck)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Gradients must flow into the hyperplane arrangement, the token encoder, and the head.
    trainable_cfg = dict(config["model"])
    trainable_cfg.pop("name", None)
    trainable_cfg.pop("packet_profile", None)
    trainable_cfg.pop("mechanism_family", None)
    trainable = build_oriented_matroid_covector_bottleneck_from_config(trainable_cfg)
    trainable.train()
    train_output = trainable(x)
    train_output["logits"].sum().backward()
    head_linear = next(m for m in trainable.head if isinstance(m, torch.nn.Linear))
    encoder_linear = next(m for m in trainable.token_encoder.net if isinstance(m, torch.nn.Linear))
    for grad in (
        trainable.arrangement.weights.grad,
        trainable.arrangement.bias.grad,
        encoder_linear.weight.grad,
        head_linear.weight.grad,
    ):
        assert grad is not None and torch.isfinite(grad).all()

    # Ablation modes must build and produce the puzzle_binary contract shape.
    for mode in (
        "magnitude_only",
        "random_hyperplanes",
        "material_role_hist_only",
        "coordinate_shuffle_by_piece",
    ):
        ablation_cfg = dict(config["model"])
        ablation_cfg.pop("name", None)
        ablation_cfg.pop("packet_profile", None)
        ablation_cfg.pop("mechanism_family", None)
        ablation_cfg["mode"] = mode
        ablation_model = build_oriented_matroid_covector_bottleneck_from_config(ablation_cfg).eval()
        with torch.no_grad():
            ablation_output = ablation_model(x)
        assert ablation_output["logits"].shape == (2,), mode
        assert torch.isfinite(ablation_output["logits"]).all(), mode

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i096"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i097_fixed_point_residual_defect_network_is_bespoke_and_conformant():
    from chess_nn_playground.models.fixed_point_residual import (
        BoardFixedPointEncoder,
        DefectTrajectoryStats,
        FixedPointResidualDefectNetwork,
        ResidualUpdateBlock,
        build_fixed_point_residual_defect_network_from_config,
    )

    folder = Path("ideas/i097_fixed_point_residual_defect_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, FixedPointResidualDefectNetwork)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "fixed_point_residual_defect_network"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES
    assert isinstance(model.encoder, BoardFixedPointEncoder)
    assert isinstance(model.shared_update, ResidualUpdateBlock)
    assert isinstance(model.stats, DefectTrajectoryStats)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — a non-trivial pin/fork pattern.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "prob",
        "defect_features",
        "h_final",
        "residual_l2",
        "residual_l1",
        "residual_cosine",
        "contraction_ratio",
        "residual_projection",
        "residual_signed_delta",
        "path_length",
        "defect_decay",
        "final_defect_l2",
        "final_defect_l1",
        "oscillation_energy",
        "defect_stats",
        "active_steps",
        "fixed_point_mode",
        "mechanism_energy",
        "proposal_profile_strength",
        "proposal_keyword_count",
    }
    assert expected_keys.issubset(output)
    steps = model.steps
    projection_dim = model.stats.projection_dim
    assert output["residual_l2"].shape == (2, steps)
    assert output["residual_l1"].shape == (2, steps)
    assert output["residual_cosine"].shape == (2, steps)
    assert output["contraction_ratio"].shape == (2, steps)
    assert output["residual_signed_delta"].shape == (2, steps)
    assert output["residual_projection"].shape == (2, steps, projection_dim)
    assert output["defect_stats"].shape == (2, 8)
    assert output["defect_features"].shape == (2, model.stats.full_output_dim)

    with torch.no_grad():
        diag_output = model(x, return_path=True)
    latent_dim = model.stats.latent_dim
    board_embed_dim = model.encoder.board_embed[0].out_features
    assert diag_output["h_path"].shape == (2, steps + 1, latent_dim)
    assert diag_output["r_path"].shape == (2, steps, latent_dim)
    assert diag_output["board_embed"].shape == (2, board_embed_dim)

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "fixed_point_residual_defect_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, FixedPointResidualDefectNetwork)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Gradients must flow into the encoder, the shared update operator, and the head.
    trainable_cfg = dict(config["model"])
    trainable_cfg.pop("name", None)
    trainable_cfg.pop("packet_profile", None)
    trainable_cfg.pop("mechanism_family", None)
    trainable = build_fixed_point_residual_defect_network_from_config(trainable_cfg)
    trainable.train()
    train_output = trainable(x)
    train_output["logits"].sum().backward()
    encoder_conv = next(
        m for m in trainable.encoder.trunk if isinstance(m, torch.nn.Conv2d)
    )
    update_linear = next(
        m for m in trainable.shared_update.net if isinstance(m, torch.nn.Linear)
    )
    head_linear = next(
        m for m in trainable.full_head if isinstance(m, torch.nn.Linear)
    )
    for grad in (
        encoder_conv.weight.grad,
        update_linear.weight.grad,
        head_linear.weight.grad,
        trainable.stats.projection.grad,
    ):
        assert grad is not None and torch.isfinite(grad).all()

    # Ablation modes must build and produce the puzzle_binary contract shape.
    for mode in (
        "final_latent_only",
        "defect_norm_only",
        "single_step",
        "untied_residual_blocks",
        "random_update_operator",
    ):
        ablation_cfg = dict(config["model"])
        ablation_cfg.pop("name", None)
        ablation_cfg.pop("packet_profile", None)
        ablation_cfg.pop("mechanism_family", None)
        ablation_cfg["mode"] = mode
        ablation_model = build_fixed_point_residual_defect_network_from_config(ablation_cfg).eval()
        with torch.no_grad():
            ablation_output = ablation_model(x)
        assert ablation_output["logits"].shape == (2,), mode
        assert torch.isfinite(ablation_output["logits"]).all(), mode

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i097"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i098_baseline_logit_residual_adapter_is_bespoke_and_conformant():
    from chess_nn_playground.models.baseline_logit_residual_adapter import (
        BaselineLogitBranch,
        BaselineLogitResidualAdapter,
        FiLMResidualAdapter,
        Simple18BoardSummary,
        build_baseline_logit_residual_adapter_from_config,
    )

    folder = Path("ideas/i098_baseline_logit_residual_adapter")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, BaselineLogitResidualAdapter)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "baseline_logit_residual_adapter"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES
    assert isinstance(model.baseline, BaselineLogitBranch)
    assert isinstance(model.summary, Simple18BoardSummary)
    assert isinstance(model.adapter, FiLMResidualAdapter)

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — a non-trivial pin/fork pattern.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "baseline_logit",
        "residual_logit",
        "adapter_correction",
        "residual_gate",
        "baseline_probability",
        "residual_to_baseline_ratio",
        "baseline_latent_norm",
        "adapter_feature_norm",
        "adapter_field_energy",
        "material_balance",
        "material_total",
        "occupancy_count",
        "rank_file_imbalance",
        "center_pressure",
        "king_ring_pressure",
    }
    assert expected_keys.issubset(output)
    assert output["baseline_logit"].shape == (2,)
    assert output["residual_logit"].shape == (2,)
    assert output["adapter_correction"].shape == (2,)
    assert output["residual_gate"].shape == (2,)
    # The decomposition s = s_b + alpha * g * s_r must hold exactly at inference.
    expected_logits = output["baseline_logit"] + output["adapter_correction"]
    assert torch.allclose(output["logits"], expected_logits, atol=1e-6)
    # The gate is a sigmoid, so it must lie in (0, 1).
    assert torch.all(output["residual_gate"] > 0.0)
    assert torch.all(output["residual_gate"] < 1.0)

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "baseline_logit_residual_adapter"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, BaselineLogitResidualAdapter)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Gradients must flow into the baseline branch, the adapter blocks, and both heads.
    trainable_cfg = dict(config["model"])
    trainable_cfg.pop("name", None)
    trainable_cfg.pop("packet_profile", None)
    trainable_cfg.pop("mechanism_family", None)
    trainable = build_baseline_logit_residual_adapter_from_config(trainable_cfg)
    trainable.train()
    train_output = trainable(x)
    train_output["logits"].sum().backward()
    baseline_conv = next(
        m for m in trainable.baseline.features.modules() if isinstance(m, torch.nn.Conv2d)
    )
    adapter_conv = next(
        m
        for m in trainable.adapter.blocks.modules()
        if isinstance(m, torch.nn.Conv2d)
    )
    baseline_head = trainable.baseline.logit_head
    residual_head_linear = next(
        m for m in trainable.adapter.residual_head if isinstance(m, torch.nn.Linear)
    )
    for grad in (
        baseline_conv.weight.grad,
        adapter_conv.weight.grad,
        baseline_head.weight.grad,
        residual_head_linear.weight.grad,
    ):
        assert grad is not None and torch.isfinite(grad).all()

    # The baseline-only ablation (residual_scale=0) must zero the adapter correction
    # without breaking the contract.
    ablation_cfg = dict(config["model"])
    ablation_cfg.pop("name", None)
    ablation_cfg.pop("packet_profile", None)
    ablation_cfg.pop("mechanism_family", None)
    ablation_cfg["residual_scale"] = 0.0
    ablation_model = build_baseline_logit_residual_adapter_from_config(ablation_cfg).eval()
    with torch.no_grad():
        ablation_output = ablation_model(x)
    assert ablation_output["logits"].shape == (2,)
    assert torch.allclose(
        ablation_output["adapter_correction"],
        torch.zeros_like(ablation_output["adapter_correction"]),
        atol=1e-7,
    )
    assert torch.allclose(ablation_output["logits"], ablation_output["baseline_logit"], atol=1e-6)

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i098"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i099_coarse_to_fine_board_residual_pyramid_is_bespoke_and_conformant():
    from chess_nn_playground.models.coarse_to_fine_residual_pyramid import (
        BoardSummary,
        CoarseToFineBoardResidualPyramid,
        PyramidResidualBlock,
        build_coarse_to_fine_board_residual_pyramid_from_config,
    )

    folder = Path("ideas/i099_coarse_to_fine_board_residual_pyramid")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert isinstance(model, CoarseToFineBoardResidualPyramid)
    assert not isinstance(model, ResearchPacketProbe)
    assert config["model"]["name"] == "coarse_to_fine_board_residual_pyramid"
    assert config["model"]["name"] not in RESEARCH_PACKET_MODEL_NAMES
    assert isinstance(model.summary, BoardSummary)
    assert any(isinstance(m, PyramidResidualBlock) for m in model.modules())

    input_channels = int(config["model"]["input_channels"])
    x = torch.zeros(2, input_channels, 8, 8)
    # White king e1, black king e8, white rook a1, white pawn e2, black queen a3,
    # black knight c3 — a non-trivial pin/fork pattern.
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 0, 6, 4] = 1.0
    x[:, 10, 5, 0] = 1.0
    x[:, 7, 5, 2] = 1.0
    x[:, 12] = 1.0

    with torch.no_grad():
        output = model(x)
    assert isinstance(output, dict)
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    expected_keys = {
        "logits",
        "coarse_l2",
        "explained_l2",
        "residual4_l1",
        "residual4_l2",
        "residual4_max",
        "residual8_l1",
        "residual8_l2",
        "residual8_max",
        "unexplained_ratio",
        "residual_gain",
        "detail_concentration",
        "residual_alignment",
        "material_balance",
        "occupancy_count",
        "rank_file_imbalance",
        "center_pressure",
    }
    assert expected_keys.issubset(output)
    # Unexplained-ratio is a normalized L2 share, must lie in [0, 1].
    assert torch.all(output["unexplained_ratio"] >= 0.0)
    assert torch.all(output["unexplained_ratio"] <= 1.0)

    fen_inputs = torch.from_numpy(
        fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ).unsqueeze(0)
    with torch.no_grad():
        fen_output = model(fen_inputs)
    assert fen_output["logits"].shape == (1,)

    # Registry-built model from the same config keeps the contract.
    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "coarse_to_fine_board_residual_pyramid"
    registry_model = build_model(registered_name, model_cfg).eval()
    assert isinstance(registry_model, CoarseToFineBoardResidualPyramid)
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert torch.isfinite(registry_output["logits"]).all()

    # Gradients must flow into the stem, the per-scale refiners, and the classifier.
    trainable_cfg = dict(config["model"])
    trainable_cfg.pop("name", None)
    trainable_cfg.pop("packet_profile", None)
    trainable_cfg.pop("mechanism_family", None)
    trainable = build_coarse_to_fine_board_residual_pyramid_from_config(trainable_cfg)
    trainable.train()
    train_output = trainable(x)
    train_output["logits"].sum().backward()
    stem_conv = next(m for m in trainable.stem.modules() if isinstance(m, torch.nn.Conv2d))
    refine4_conv = next(m for m in trainable.residual4_refine.modules() if isinstance(m, torch.nn.Conv2d))
    refine8_conv = next(m for m in trainable.residual8_refine.modules() if isinstance(m, torch.nn.Conv2d))
    classifier_linear = trainable.classifier
    for grad in (
        stem_conv.weight.grad,
        refine4_conv.weight.grad,
        refine8_conv.weight.grad,
        classifier_linear.weight.grad,
    ):
        assert grad is not None and torch.isfinite(grad).all()

    # The idea folder must not depend on the shared ResearchPacketProbe scaffold.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called
    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i099"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i103_attention_disagreement_residual_network_is_bespoke_and_conformant():
    folder = Path("ideas/i103_attention_disagreement_residual_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0   # white king e1
    x[:, 11, 0, 4] = 1.0  # black king e8
    x[:, 3, 7, 0] = 1.0   # white rook a1
    x[:, 10, 0, 0] = 1.0  # black queen a8
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "attention",
        "attended_values",
        "disagreement_features",
        "attention_js_divergence_mean",
        "attention_js_divergence_max",
        "attention_entropy_variance",
        "family_map_cosine_distance_max",
        "query_map_cosine_distance_max",
        "attended_residual_norm",
        "attended_covariance_trace",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    family_count = int(config["model"].get("family_count", 4))
    query_count = int(config["model"].get("query_count", 8))
    assert output["attention"].shape == (2, family_count, query_count, 64)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "attention_disagreement_residual_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i103"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i104_cross_scale_attention_residual_network_is_bespoke_and_conformant():
    folder = Path("ideas/i104_cross_scale_attention_residual_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 0, 0] = 1.0
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "attention_actual",
        "attention_predicted",
        "residual_attention",
        "fine_to_coarse_attention",
        "coarse_to_fine_attention",
        "residual_features",
        "residual_total_energy",
        "residual_off_diagonal_energy",
        "residual_max_abs",
        "residual_self_diagonal_mean",
        "residual_asymmetry",
        "residual_frobenius",
        "residual_row_entropy_mean",
        "residual_row_entropy_variance",
        "residual_per_source_l1",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())
    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["attention_actual"].shape == (2, 64, 64)
    assert output["attention_predicted"].shape == (2, 64, 64)
    assert output["residual_attention"].shape == (2, 64, 64)
    coarse_scale = int(config["model"].get("coarse_scale", 2))
    num_coarse = (8 // coarse_scale) ** 2
    assert output["fine_to_coarse_attention"].shape == (2, 64, num_coarse)
    assert output["coarse_to_fine_attention"].shape == (2, num_coarse, 64)
    assert output["residual_per_source_l1"].shape == (2, 64)
    assert torch.allclose(output["attention_actual"].sum(dim=-1), torch.ones(2, 64), atol=1.0e-5)
    assert torch.allclose(output["attention_predicted"].sum(dim=-1), torch.ones(2, 64), atol=1.0e-5)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "cross_scale_attention_residual_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i104"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i105_slot_attention_role_binding_network_is_bespoke_and_conformant():
    folder = Path("ideas/i105_slot_attention_role_binding_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 0, 0] = 1.0
    x[:, 0, 1, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "slots",
        "assignments",
        "slot_updates",
        "update_residuals",
        "slot_mass",
        "slot_share",
        "slot_self_entropy",
        "per_token_entropy",
        "mean_token_entropy",
        "token_entropy_variance",
        "slot_norms",
        "slot_dispersion",
        "token_mask",
        "occupancy_mask",
        "diagnostic_features",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    num_slots = int(config["model"].get("num_slots", 8))
    num_iterations = int(config["model"].get("num_iterations", 3))
    max_tokens = int(config["model"].get("max_tokens", 32))

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["slots"].shape[0] == 2
    assert output["slots"].shape[1] == num_slots
    assert output["assignments"].shape == (2, num_iterations, num_slots, max_tokens)
    assert output["slot_updates"].shape[0] == 2
    assert output["slot_updates"].shape[1] == num_iterations
    assert output["slot_updates"].shape[2] == num_slots
    assert output["update_residuals"].shape == (2, num_iterations)
    assert output["slot_mass"].shape == (2, num_slots)
    assert output["token_mask"].shape == (2, max_tokens)

    final_assignment = output["assignments"][:, -1]
    assert torch.allclose(
        final_assignment.sum(dim=1),
        output["token_mask"],
        atol=1.0e-5,
    )
    expected_total_mass = output["token_mask"].sum(dim=-1)
    assert torch.allclose(output["slot_mass"].sum(dim=-1), expected_total_mass, atol=1.0e-5)
    assert (output["slot_mass"] >= -1.0e-6).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "slot_attention_role_binding_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i105"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i106_attention_perturbation_sensitivity_network_is_bespoke_and_conformant():
    folder = Path("ideas/i106_attention_perturbation_sensitivity_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 0, 0] = 1.0
    x[:, 0, 1, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "latent_base",
        "latent_top",
        "latent_low",
        "latent_random",
        "latent_neighborhood",
        "attention",
        "per_square_attention",
        "mean_query_entropy",
        "max_attention",
        "topk_attention_mass",
        "attention_occupied_mass",
        "attention_empty_mass",
        "attention_query_disagreement",
        "attention_range",
        "occupancy_mask",
        "keep_mask_top",
        "keep_mask_low",
        "keep_mask_random",
        "keep_mask_neighborhood",
        "delta_top",
        "delta_low",
        "delta_random",
        "delta_neighborhood",
        "contrast_top_minus_low",
        "contrast_top_minus_random",
        "contrast_neighborhood_minus_top",
        "ratio_top_over_low",
        "sensitivity_features",
        "attention_features",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    num_queries = int(config["model"].get("num_queries", 8))
    top_k = int(config["model"].get("top_k", 6))
    token_dim = int(config["model"].get("token_dim", config["model"].get("channels", 64)))

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["latent_base"].shape == (2, token_dim)
    assert output["latent_top"].shape == (2, token_dim)
    assert output["latent_low"].shape == (2, token_dim)
    assert output["latent_random"].shape == (2, token_dim)
    assert output["latent_neighborhood"].shape == (2, token_dim)
    assert output["attention"].shape == (2, num_queries, 64)
    assert output["per_square_attention"].shape == (2, 64)
    assert output["occupancy_mask"].shape == (2, 64)
    for key in (
        "keep_mask_top",
        "keep_mask_low",
        "keep_mask_random",
        "keep_mask_neighborhood",
    ):
        assert output[key].shape == (2, 64)
    for key in (
        "delta_top",
        "delta_low",
        "delta_random",
        "delta_neighborhood",
        "contrast_top_minus_low",
        "contrast_top_minus_random",
        "contrast_neighborhood_minus_top",
        "ratio_top_over_low",
    ):
        assert output[key].shape == (2,)
    assert output["sensitivity_features"].shape == (2, 8)
    assert output["attention_features"].shape == (2, 7)

    assert torch.allclose(
        output["attention"].sum(dim=-1), torch.ones(2, num_queries), atol=1.0e-5
    )
    assert torch.allclose(
        output["per_square_attention"].sum(dim=-1), torch.ones(2), atol=1.0e-5
    )
    # Each keep mask must zero exactly top_k squares (no more, no fewer) for the
    # top, low, and random selections; the neighbourhood mask zeros 1..9 squares.
    for key in ("keep_mask_top", "keep_mask_low", "keep_mask_random"):
        zeros_per_sample = (output[key] < 0.5).sum(dim=-1)
        assert (zeros_per_sample == top_k).all(), key
    nbhd_zeros = (output["keep_mask_neighborhood"] < 0.5).sum(dim=-1)
    assert (nbhd_zeros >= 1).all()
    assert (nbhd_zeros <= 9).all()
    # Sensitivity scalars are non-negative norms.
    for key in ("delta_top", "delta_low", "delta_random", "delta_neighborhood"):
        assert (output[key] >= -1.0e-6).all(), key
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "attention_perturbation_sensitivity_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i106"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i107_kernel_mean_prototype_network_is_bespoke_and_conformant():
    folder = Path("ideas/i107_kernel_mean_prototype_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)
    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 0, 0] = 1.0
    x[:, 0, 1, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "kernel_mean",
        "kernel_features",
        "occupancy_mask",
        "occupied_count",
        "log_occupied_count",
        "canonical_piece_counts",
        "us_them_imbalance",
        "kernel_self_similarity",
        "prototype_distances",
        "prototype_similarities",
        "prototype_log_gamma",
        "diagnostic_features",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    token_dim = int(config["model"].get("token_dim", config["model"].get("channels", 64)))
    phi_dim = int(config["model"].get("phi_dim", config["model"].get("kernel_dim", 128)))
    num_prototypes = int(config["model"].get("num_prototypes", 8))

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["kernel_mean"].shape == (2, phi_dim)
    assert output["kernel_features"].shape == (2, 64, phi_dim)
    assert output["occupancy_mask"].shape == (2, 64)
    assert output["occupied_count"].shape == (2,)
    assert output["log_occupied_count"].shape == (2,)
    assert output["canonical_piece_counts"].shape == (2, 6)
    assert output["us_them_imbalance"].shape == (2,)
    assert output["kernel_self_similarity"].shape == (2,)
    assert output["prototype_distances"].shape == (2, num_prototypes)
    assert output["prototype_similarities"].shape == (2, num_prototypes)
    assert output["prototype_log_gamma"].shape == (2, num_prototypes)
    diagnostic_dim = 1 + 6 + 1 + 1 + 2 * num_prototypes
    assert output["diagnostic_features"].shape == (2, diagnostic_dim)

    assert (output["occupied_count"] == 5).all()
    assert (output["prototype_distances"] >= -1.0e-6).all()
    assert (output["prototype_similarities"] >= 0.0).all()
    assert (output["prototype_similarities"] <= 1.0 + 1.0e-6).all()
    assert (output["kernel_self_similarity"] >= -1.0e-6).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "kernel_mean_prototype_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i107"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i108_tensorsketch_interaction_network_is_bespoke_and_conformant():
    folder = Path("ideas/i108_tensorsketch_interaction_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    x = torch.zeros(2, int(config["model"]["input_channels"]), 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 0, 0] = 1.0
    x[:, 0, 1, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "board_feature_vector",
        "base_features",
        "count_sketch",
        "tensor_sketches",
        "sketch_means",
        "sketch_energies",
        "base_mean",
        "base_energy",
        "diagnostic_features",
        "degree_log_scale",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    base_dim = int(config["model"].get("base_dim", config["model"].get("hidden_dim", 512)))
    sketch_dim = int(
        config["model"].get("sketch_dim", config["model"].get("base_dim", config["model"].get("hidden_dim", 512)))
    )
    sketch_degrees = tuple(config["model"].get("sketch_degrees", (2, 3)))
    if not isinstance(sketch_degrees, tuple):
        sketch_degrees = tuple(sketch_degrees)

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["base_features"].shape == (2, base_dim)
    assert output["count_sketch"].shape == (2, sketch_dim)
    assert output["tensor_sketches"].shape == (2, len(sketch_degrees), sketch_dim)
    assert output["sketch_means"].shape == (2, len(sketch_degrees))
    assert output["sketch_energies"].shape == (2, len(sketch_degrees))
    assert output["base_mean"].shape == (2,)
    assert output["base_energy"].shape == (2,)
    assert output["diagnostic_features"].shape == (2, 2 + 2 * len(sketch_degrees))
    assert output["degree_log_scale"].shape == (2, len(sketch_degrees))
    assert (output["base_energy"] >= -1.0e-6).all()
    assert (output["sketch_energies"] >= -1.0e-6).all()
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # CountSketch hashes/signs are frozen so the count sketch is deterministic across runs.
    with torch.no_grad():
        repeat = model(x)
    assert torch.allclose(output["count_sketch"], repeat["count_sketch"])

    model_cfg = dict(config["model"])
    registered_name = model_cfg.pop("name")
    assert registered_name == "tensorsketch_interaction_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i108"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i109_maxout_region_signature_network_is_bespoke_and_conformant():
    folder = Path("ideas/i109_maxout_region_signature_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    model_cfg = dict(config["model"])
    input_channels = int(model_cfg["input_channels"])
    channels = int(model_cfg.get("channels", 64))
    bank_units = int(model_cfg.get("bank_units", 8))
    bank_pieces = int(model_cfg.get("bank_pieces", 4))
    num_banks = int(model_cfg.get("num_banks", 2))

    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 0, 0] = 1.0
    x[:, 0, 1, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "trunk_pool",
        "signature_vector",
        "bank_activations",
        "winner_histograms",
        "region_counts",
        "rank_region_counts",
        "file_region_counts",
        "horizontal_transitions",
        "vertical_transitions",
        "margin_stats",
        "activation_stats",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["trunk_pool"].shape == (2, channels)
    assert output["bank_activations"].shape == (2, num_banks, bank_units, 8, 8)
    assert output["winner_histograms"].shape == (2, num_banks, bank_units, bank_pieces)
    assert output["region_counts"].shape == (2, num_banks, bank_units)
    assert output["rank_region_counts"].shape == (2, num_banks, bank_units)
    assert output["file_region_counts"].shape == (2, num_banks, bank_units)
    assert output["horizontal_transitions"].shape == (2, num_banks, bank_units)
    assert output["vertical_transitions"].shape == (2, num_banks, bank_units)
    assert output["margin_stats"].shape == (2, num_banks, bank_units, 4)
    assert output["activation_stats"].shape == (2, num_banks, bank_units, 4)

    sig_dim_per_bank = (
        bank_units * bank_pieces  # winner histogram
        + bank_units  # region count
        + 2 * bank_units  # rank/file region counts
        + 2 * bank_units  # horizontal / vertical transitions
        + 4 * bank_units  # margin stats
        + 4 * bank_units  # activation stats
    )
    assert output["signature_vector"].shape == (2, num_banks * sig_dim_per_bank)

    # Region statistics must be in their natural ranges.
    assert (output["winner_histograms"] >= 0).all()
    hist_sums = output["winner_histograms"].sum(dim=-1)
    assert torch.allclose(hist_sums, torch.ones_like(hist_sums), atol=1.0e-5)
    assert (output["region_counts"] >= 1).all()
    assert (output["region_counts"] <= bank_pieces).all()
    assert (output["horizontal_transitions"] >= 0).all()
    assert (output["horizontal_transitions"] <= 8 * 7).all()
    assert (output["vertical_transitions"] >= 0).all()
    assert (output["vertical_transitions"] <= 8 * 7).all()
    # Margin top1-top2 is non-negative by construction.
    assert (output["margin_stats"][..., 2] >= -1.0e-6).all()  # max
    assert (output["margin_stats"][..., 3] >= -1.0e-6).all()  # min

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    registered_name = model_cfg.pop("name")
    assert registered_name == "maxout_region_signature_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i109"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i110_spline_board_surface_network_is_bespoke_and_conformant():
    folder = Path("ideas/i110_spline_board_surface_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    model_cfg = dict(config["model"])
    input_channels = int(model_cfg["input_channels"])
    spline_basis_size = int(model_cfg.get("spline_basis_size", 4))
    residual_summary_channels = int(model_cfg.get("residual_summary_channels", 32))
    num_basis = spline_basis_size * spline_basis_size

    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 0, 0] = 1.0
    x[:, 0, 1, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "coefficients",
        "reconstruction",
        "residuals",
        "residual_energy",
        "residual_summary_mean",
        "residual_summary_max",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["coefficients"].shape == (2, input_channels, num_basis)
    assert output["reconstruction"].shape == (2, input_channels, 8, 8)
    assert output["residuals"].shape == (2, input_channels, 8, 8)
    assert output["residual_energy"].shape == (2, input_channels)
    assert output["residual_summary_mean"].shape == (2, residual_summary_channels)
    assert output["residual_summary_max"].shape == (2, residual_summary_channels)

    # The smooth fit + residual must reconstruct the input exactly.
    rebuilt = output["reconstruction"] + output["residuals"]
    assert torch.allclose(rebuilt, x, atol=1.0e-4)

    # Residual energies are non-negative by construction.
    assert (output["residual_energy"] >= -1.0e-6).all()

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    registered_name = model_cfg.pop("name")
    assert registered_name == "spline_board_surface_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i110"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i111_boundary_condition_disagreement_cnn_is_bespoke_and_conformant():
    folder = Path("ideas/i111_boundary_condition_disagreement_cnn")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    model_cfg = dict(config["model"])
    input_channels = int(model_cfg["input_channels"])
    channels = int(model_cfg.get("channels", 64))
    boundary_modes = tuple(model.boundary_modes)
    num_modes = len(boundary_modes)
    assert num_modes >= 2
    assert set(boundary_modes).issubset({"zeros", "reflect", "replicate", "circular"})

    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0
    x[:, 3, 7, 0] = 1.0
    x[:, 10, 0, 0] = 1.0
    x[:, 0, 1, 4] = 1.0
    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "boundary_features",
        "disagreement_map",
        "disagreement_mean",
        "disagreement_max",
        "disagreement_energy",
        "pairwise_disagreement_energy",
        "per_mode_mean",
        "per_mode_max",
        "per_mode_pooled",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["boundary_features"].shape == (num_modes, 2, channels, 8, 8)
    assert output["disagreement_map"].shape == (2, channels, 8, 8)
    assert output["disagreement_mean"].shape == (2, channels)
    assert output["disagreement_max"].shape == (2, channels)
    assert output["disagreement_energy"].shape == (2, 2 * channels)
    assert output["pairwise_disagreement_energy"].shape == (2, num_modes, num_modes)
    assert output["per_mode_mean"].shape == (2, num_modes, channels)
    assert output["per_mode_max"].shape == (2, num_modes, channels)
    assert output["per_mode_pooled"].shape == (2, num_modes, 2 * channels)

    # Variance and pairwise energies are non-negative.
    assert (output["disagreement_map"] >= -1.0e-6).all()
    assert (output["pairwise_disagreement_energy"] >= -1.0e-6).all()
    diag = torch.diagonal(output["pairwise_disagreement_energy"], dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1.0e-5)
    pairwise = output["pairwise_disagreement_energy"]
    assert torch.allclose(pairwise, pairwise.transpose(-1, -2), atol=1.0e-5)

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    registered_name = model_cfg.pop("name")
    assert registered_name == "boundary_condition_disagreement_cnn"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i111"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i112_piece_drop_stability_network_is_bespoke_and_conformant():
    folder = Path("ideas/i112_piece_drop_stability_network")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    model_cfg = dict(config["model"])
    input_channels = int(model_cfg["input_channels"])
    latent_dim = int(model_cfg.get("latent_dim", model_cfg.get("hidden_dim", 64)))
    drop_masks = tuple(model.drop_masks)
    num_masks = len(drop_masks)
    assert num_masks >= 4
    assert set(drop_masks).issubset(
        {"own_minor", "own_major", "opp_minor", "opp_major", "center", "king_neigh"}
    )

    x = torch.zeros(2, input_channels, 8, 8)
    # Side-to-move = white for first sample, black for second.
    x[0, 12] = 1.0
    x[1, 12] = 0.0
    # Place pieces relevant to every mask group.
    x[:, 5, 7, 4] = 1.0   # white king
    x[:, 11, 0, 4] = 1.0  # black king
    x[:, 1, 6, 5] = 1.0   # white knight (own minor / king_neigh near white king)
    x[:, 8, 1, 5] = 1.0   # black bishop (opponent minor)
    x[:, 3, 7, 0] = 1.0   # white rook (own major)
    x[:, 9, 0, 0] = 1.0   # black rook (opponent major)
    x[:, 0, 3, 3] = 1.0   # white pawn on d5 (center)
    x[:, 6, 4, 4] = 1.0   # black pawn on e4 (center)

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "original_latent",
        "masked_latents",
        "delta_vectors",
        "stability",
        "stability_ratio",
        "original_norm",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["original_latent"].shape == (2, latent_dim)
    assert output["masked_latents"].shape == (2, num_masks, latent_dim)
    assert output["delta_vectors"].shape == (2, num_masks, latent_dim)
    assert output["stability"].shape == (2, num_masks)
    assert output["stability_ratio"].shape == (2, num_masks)
    assert output["original_norm"].shape == (2,)

    # Stability magnitudes are non-negative and consistent with the delta vector norms.
    assert (output["stability"] >= -1.0e-6).all()
    assert (output["stability_ratio"] >= -1.0e-6).all()
    assert torch.allclose(
        output["stability"],
        output["delta_vectors"].norm(dim=-1),
        atol=1.0e-5,
    )

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # At least one mask should produce a non-zero latent shift on this packed board.
    assert output["stability"].sum() > 0.0

    registered_name = model_cfg.pop("name")
    assert registered_name == "piece_drop_stability_network"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i112"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i113_row_file_factor_mixer_is_bespoke_and_conformant():
    folder = Path("ideas/i113_row_file_factor_mixer")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    model_cfg = dict(config["model"])
    input_channels = int(model_cfg["input_channels"])
    channels = int(model_cfg["channels"])
    depth = int(model_cfg["depth"])
    assert depth >= 1

    x = torch.zeros(2, input_channels, 8, 8)
    if input_channels > 12:
        x[0, 12] = 1.0
    # Place a few pieces so rank/file/bilinear energies are non-trivial.
    x[:, 0, 1, 4] = 1.0
    x[:, 6, 6, 3] = 1.0
    x[:, 5, 7, 4] = 1.0
    x[:, 11, 0, 4] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "pooled_features",
        "rank_summary",
        "file_summary",
        "rank_energy",
        "file_energy",
        "bilinear_energy",
        "rank_energy_per_block",
        "file_energy_per_block",
        "bilinear_energy_per_block",
        "rank_file_imbalance",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["pooled_features"].shape == (2, channels)
    assert output["rank_summary"].shape == (2, channels, 8)
    assert output["file_summary"].shape == (2, channels, 8)
    assert output["rank_energy"].shape == (2,)
    assert output["file_energy"].shape == (2,)
    assert output["bilinear_energy"].shape == (2,)
    assert output["rank_energy_per_block"].shape == (2, depth)
    assert output["file_energy_per_block"].shape == (2, depth)
    assert output["bilinear_energy_per_block"].shape == (2, depth)
    assert output["rank_file_imbalance"].shape == (2,)

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Energies are non-negative (squared-activation means).
    assert (output["rank_energy"] >= -1.0e-6).all()
    assert (output["file_energy"] >= -1.0e-6).all()
    assert (output["bilinear_energy"] >= -1.0e-6).all()
    assert (output["rank_file_imbalance"] >= -1.0e-6).all()
    assert (output["rank_file_imbalance"] <= 1.0 + 1.0e-6).all()
    # Per-block rank energies must sum to the aggregate rank energy.
    assert torch.allclose(
        output["rank_energy"],
        output["rank_energy_per_block"].sum(dim=-1),
        atol=1.0e-5,
    )
    # The packed board should drive at least one factor to a non-zero
    # energy in expectation; in particular the total mixer energy should
    # be strictly positive.
    total_mixer_energy = (
        output["rank_energy"] + output["file_energy"] + output["bilinear_energy"]
    )
    assert total_mixer_energy.sum() > 0.0

    registered_name = model_cfg.pop("name")
    assert registered_name == "row_file_factor_mixer"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i113"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i114_piece_conditioned_hypernetwork_cnn_is_bespoke_and_conformant():
    folder = Path("ideas/i114_piece_conditioned_hypernetwork_cnn")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    model_cfg = dict(config["model"])
    input_channels = int(model_cfg["input_channels"])
    channels = int(model_cfg["channels"])
    depth = int(model_cfg["depth"])
    assert depth >= 1

    # Two boards with different piece inventories so the hypernetwork
    # must produce different per-sample gates and kernels.
    x = torch.zeros(2, input_channels, 8, 8)
    if input_channels > 12:
        x[0, 12] = 1.0
    # Sample 0: dense own-side material on the back rank.
    x[0, 0, 0, 0] = 1.0  # white pawn
    x[0, 1, 0, 1] = 1.0  # white knight
    x[0, 2, 0, 2] = 1.0  # white bishop
    x[0, 3, 0, 3] = 1.0  # white rook
    x[0, 4, 0, 4] = 1.0  # white queen
    x[0, 5, 0, 5] = 1.0  # white king
    # Sample 1: only black pieces, sparse.
    x[1, 6, 7, 0] = 1.0  # black pawn
    x[1, 11, 7, 4] = 1.0  # black king

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "pooled_features",
        "inventory_summary",
        "gate_mean",
        "gate_entropy",
        "kernel_energy",
        "gated_energy",
        "gate_mean_per_block",
        "gate_entropy_per_block",
        "kernel_norm_per_block",
        "block_energy_per_block",
        "material_delta",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["pooled_features"].shape == (2, channels)
    assert output["inventory_summary"].shape == (2, 27)
    assert output["gate_mean"].shape == (2,)
    assert output["gate_entropy"].shape == (2,)
    assert output["kernel_energy"].shape == (2,)
    assert output["gated_energy"].shape == (2,)
    assert output["gate_mean_per_block"].shape == (2, depth)
    assert output["gate_entropy_per_block"].shape == (2, depth)
    assert output["kernel_norm_per_block"].shape == (2, depth)
    assert output["block_energy_per_block"].shape == (2, depth)
    assert output["material_delta"].shape == (2,)

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Gates pass through sigmoid so they live in [0, 1].
    assert (output["gate_mean_per_block"] >= -1.0e-6).all()
    assert (output["gate_mean_per_block"] <= 1.0 + 1.0e-6).all()
    # Kernel norms and post-activation block energies are non-negative.
    assert (output["kernel_norm_per_block"] >= -1.0e-6).all()
    assert (output["block_energy_per_block"] >= -1.0e-6).all()
    # Per-block kernel norms must sum to the aggregate kernel energy.
    assert torch.allclose(
        output["kernel_energy"],
        output["kernel_norm_per_block"].sum(dim=-1),
        atol=1.0e-5,
    )
    # Per-block block energies must sum to the aggregate gated energy.
    assert torch.allclose(
        output["gated_energy"],
        output["block_energy_per_block"].sum(dim=-1),
        atol=1.0e-5,
    )

    # Different piece inventories must produce different summaries.
    summary_diff = (output["inventory_summary"][0] - output["inventory_summary"][1]).abs().sum()
    assert summary_diff.item() > 0.0
    # And different summaries must yield different per-sample gates.
    gate_diff = (
        output["gate_mean_per_block"][0] - output["gate_mean_per_block"][1]
    ).abs().sum()
    assert gate_diff.item() > 0.0

    # The dense, own-material sample must have a strictly positive
    # material delta and the bare-king sample must have a strictly
    # negative one.
    assert output["material_delta"][0].item() > 0.0
    assert output["material_delta"][1].item() < 0.0

    # Total post-activation energy must be strictly positive: a static
    # all-zeros depthwise kernel would not yield this.
    assert output["gated_energy"].sum() > 0.0

    registered_name = model_cfg.pop("name")
    assert registered_name == "piece_conditioned_hypernetwork_cnn"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i114"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i115_neural_board_cellular_automaton_is_bespoke_and_conformant():
    folder = Path("ideas/i115_neural_board_cellular_automaton")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    model_cfg = dict(config["model"])
    input_channels = int(model_cfg["input_channels"])
    channels = int(model_cfg["channels"])
    steps = int(model_cfg.get("steps", 6))
    assert steps >= 1

    x = torch.zeros(2, input_channels, 8, 8)
    if input_channels > 12:
        x[0, 12] = 1.0
    # Sample 0: a complex pattern likely to require relaxation.
    x[0, 0, 1, 4] = 1.0
    x[0, 1, 0, 1] = 1.0
    x[0, 5, 0, 4] = 1.0
    x[0, 11, 7, 4] = 1.0
    x[0, 6, 6, 3] = 1.0
    # Sample 1: a sparser pattern.
    x[1, 5, 7, 4] = 1.0
    x[1, 11, 0, 4] = 1.0

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "pooled_features",
        "final_state",
        "update_energy",
        "update_energy_mean",
        "final_step_update_energy",
        "update_energy_per_step",
        "state_energy_per_step",
        "final_state_energy",
        "step_size",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["pooled_features"].shape == (2, channels)
    assert output["final_state"].shape == (2, channels, 8, 8)
    assert output["update_energy"].shape == (2,)
    assert output["update_energy_mean"].shape == (2,)
    assert output["final_step_update_energy"].shape == (2,)
    assert output["update_energy_per_step"].shape == (2, steps)
    assert output["state_energy_per_step"].shape == (2, steps + 1)
    assert output["final_state_energy"].shape == (2,)
    assert output["step_size"].shape == (2,)

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # Energies are squared norms divided by N, so they are non-negative.
    assert (output["update_energy_per_step"] >= -1.0e-6).all()
    assert (output["state_energy_per_step"] >= -1.0e-6).all()
    assert (output["update_energy"] >= -1.0e-6).all()
    assert (output["final_state_energy"] >= -1.0e-6).all()

    # Per-step update energies must sum to the aggregate update energy.
    assert torch.allclose(
        output["update_energy"],
        output["update_energy_per_step"].sum(dim=-1),
        atol=1.0e-5,
    )
    # Per-step update mean must equal mean over the steps axis.
    assert torch.allclose(
        output["update_energy_mean"],
        output["update_energy_per_step"].mean(dim=-1),
        atol=1.0e-5,
    )
    # final_step_update_energy must equal the last column of the trajectory.
    assert torch.allclose(
        output["final_step_update_energy"],
        output["update_energy_per_step"][:, -1],
        atol=1.0e-5,
    )

    # The learnable step size is sigmoid-bounded by max_step_size (default 1.0).
    max_step_size = float(model_cfg.get("max_step_size", 1.0))
    assert (output["step_size"] >= 0.0).all()
    assert (output["step_size"] <= max_step_size + 1.0e-6).all()

    # The update rule's final 1x1 conv is zero-initialized, so an
    # untrained model should produce numerically zero per-step update
    # energies (the CA dynamics start out as the identity fixed point).
    assert torch.allclose(
        output["update_energy_per_step"],
        torch.zeros_like(output["update_energy_per_step"]),
        atol=1.0e-7,
    )

    # The state energy at every step should equal the initial state
    # energy (no update has fired) at init.
    initial_state_energy = output["state_energy_per_step"][:, :1]
    assert torch.allclose(
        output["state_energy_per_step"],
        initial_state_energy.expand_as(output["state_energy_per_step"]),
        atol=1.0e-5,
    )

    # The CA must apply the *same* local update rule at every step; the
    # bespoke model exposes a single _LocalUpdateRule submodule that
    # is reused across iterations.
    update_rule_modules = [m for m in model.modules() if type(m).__name__ == "_LocalUpdateRule"]
    assert len(update_rule_modules) == 1, (
        "Tied-weight CA must expose exactly one shared local update rule, "
        f"got {len(update_rule_modules)}"
    )

    # Once the update rule's final 1x1 conv has non-zero weights, the
    # per-step update energy must be strictly positive: this confirms
    # the iteration loop actually fires `f` at each step.
    with torch.no_grad():
        model.update_rule.update.weight.normal_(mean=0.0, std=0.05)
        if model.update_rule.update.bias is not None:
            model.update_rule.update.bias.normal_(mean=0.0, std=0.05)
    with torch.no_grad():
        active_output = model(x)
    assert (active_output["update_energy_per_step"] > 0.0).all()
    assert active_output["update_energy"].sum() > 0.0
    # And the final state must differ from the initial embedded state.
    embed_state = model.embed(x)
    assert not torch.allclose(active_output["final_state"], embed_state, atol=1.0e-5)

    registered_name = model_cfg.pop("name")
    assert registered_name == "neural_board_cellular_automaton"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i115"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues


def test_i116_symmetric_difference_twin_encoder_is_bespoke_and_conformant():
    folder = Path("ideas/i116_symmetric_difference_twin_encoder")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    module = _load_idea_model(folder)
    model = module.build_model_from_config(config).eval()

    assert not isinstance(model, ResearchPacketProbe)

    model_cfg = dict(config["model"])
    input_channels = int(model_cfg["input_channels"])
    channels = int(model_cfg["channels"])
    hidden_dim = int(model_cfg["hidden_dim"])

    # Build a non-trivial board pair so the safe transform changes the
    # input (the file mirror is a no-op on a board that is symmetric
    # under file flip plus castling-channel swap).
    x = torch.zeros(2, input_channels, 8, 8)
    x[:, 12] = 1.0  # white to move
    x[0, 0, 1, 0] = 1.0  # white pawn on a2
    x[0, 5, 0, 4] = 1.0  # white king on e1
    x[0, 11, 7, 4] = 1.0  # black king on e8
    x[0, 13, :, :] = 1.0  # white kingside castling rights
    x[1, 0, 1, 7] = 1.0  # white pawn on h2
    x[1, 5, 0, 6] = 1.0  # white king on g1
    x[1, 11, 7, 4] = 1.0  # black king on e8
    x[1, 14, :, :] = 1.0  # white queenside castling rights

    with torch.no_grad():
        output = model(x)

    expected_keys = {
        "logits",
        "pooled_preserved",
        "pooled_changed",
        "pooled_fused",
        "preserved_map",
        "changed_map",
        "fused_map",
        "z",
        "z_transformed",
        "symmetric_difference_energy",
        "preserved_energy",
        "latent_disagreement",
        "symmetry_residual",
    }
    assert isinstance(output, dict)
    assert expected_keys.issubset(output.keys())

    assert output["logits"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert output["pooled_preserved"].shape == (2, channels)
    assert output["pooled_changed"].shape == (2, channels)
    assert output["pooled_fused"].shape == (2, hidden_dim)
    assert output["preserved_map"].shape == (2, channels, 8, 8)
    assert output["changed_map"].shape == (2, channels, 8, 8)
    assert output["fused_map"].shape == (2, hidden_dim, 8, 8)
    assert output["z"].shape == (2, channels, 8, 8)
    assert output["z_transformed"].shape == (2, channels, 8, 8)
    assert output["symmetric_difference_energy"].shape == (2,)
    assert output["preserved_energy"].shape == (2,)
    assert output["latent_disagreement"].shape == (2,)
    assert output["symmetry_residual"].shape == (2,)

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key

    # changed = |z - z_aligned| must be elementwise non-negative.
    assert (output["changed_map"] >= 0.0).all()
    # symmetric_difference_energy is a squared norm so it must be >= 0.
    assert (output["symmetric_difference_energy"] >= -1.0e-6).all()
    assert (output["preserved_energy"] >= -1.0e-6).all()
    assert (output["latent_disagreement"] >= -1.0e-6).all()
    # The element-wise mean of |z - z_aligned| equals symmetry_residual.
    assert torch.allclose(
        output["symmetry_residual"],
        output["changed_map"].mean(dim=(1, 2, 3)),
        atol=1.0e-6,
    )
    # preserved + changed reconstructs the larger of z and z_aligned per sign:
    #   z + z_aligned = 2 * preserved
    #   |z - z_aligned| = changed
    z_plus = output["z"] + output["z_transformed"]
    assert torch.allclose(2.0 * output["preserved_map"], z_plus, atol=1.0e-5)
    assert torch.allclose(
        (output["z"] - output["z_transformed"]).abs(), output["changed_map"], atol=1.0e-6
    )

    # The model must expose exactly one shared trunk module reused for
    # both branches (twin pass with tied weights). Untying would defeat
    # the point of the architecture.
    trunk_modules = [m for m in model.modules() if type(m).__name__ == "_SharedBoardTrunk"]
    assert len(trunk_modules) == 1, (
        "Twin encoder must expose exactly one shared trunk module, "
        f"got {len(trunk_modules)}"
    )

    # The safe transform must actually transform the inputs in this
    # batch (otherwise the comparison is vacuous).
    x_t = model.safe_transform(x)
    assert not torch.allclose(x_t, x)
    # And it must be an involution.
    assert torch.allclose(model.safe_transform(x_t), x)

    # The transform must permute the kingside / queenside castling
    # channels (planes 13/14 and 15/16) on simple_18.
    src = torch.zeros(1, input_channels, 8, 8)
    src[0, 13] = 1.0
    flipped = model.safe_transform(src)
    assert flipped[0, 14].sum() > 0.0
    assert flipped[0, 13].sum() == 0.0
    src2 = torch.zeros(1, input_channels, 8, 8)
    src2[0, 15] = 1.0
    flipped2 = model.safe_transform(src2)
    assert flipped2[0, 16].sum() > 0.0
    assert flipped2[0, 15].sum() == 0.0

    # If the inputs differ across the file flip, the changed_map must
    # have non-zero energy somewhere; otherwise the model has nothing
    # to read.
    assert output["symmetric_difference_energy"].sum() > 0.0

    # Building from the registry must work and must not be backed by
    # the research-packet probe.
    registered_name = model_cfg.pop("name")
    assert registered_name == "symmetric_difference_twin_encoder"
    registry_model = build_model(registered_name, model_cfg).eval()
    with torch.no_grad():
        registry_output = registry_model(x)
    assert registry_output["logits"].shape == (2,)
    assert registered_name not in RESEARCH_PACKET_MODEL_NAMES

    # The idea-local wrapper must not import or call the research packet
    # probe.
    wiring = analyze_model_wiring(folder / "model.py")
    forbidden = {"ResearchPacketProbe", "build_research_packet_probe_from_config"}
    imported = {item.rsplit(".", 1)[-1] for item in wiring.imports}
    called = {item.rsplit(".", 1)[-1] for item in wiring.calls}
    assert not (imported & forbidden)
    assert "build_research_packet_probe_from_config" not in called

    model_py = (folder / "model.py").read_text(encoding="utf-8")
    assert "ResearchPacketProbe" not in model_py
    assert "build_research_packet_probe_from_config" not in model_py

    kind_row = detect_idea_implementation_kind(folder)
    assert kind_row.detected_kind == "bespoke_model"
    assert kind_row.implementation_status == "implemented"
    assert not kind_row.issues

    training_report = validate_idea_for_training(folder)
    assert training_report["valid"], training_report

    conformance_rows = [row for row in _audit_architecture_conformance_rows() if row.idea_id == "i116"]
    assert len(conformance_rows) == 1
    assert conformance_rows[0].implementation_kind == "bespoke_model"
    assert not conformance_rows[0].issues
