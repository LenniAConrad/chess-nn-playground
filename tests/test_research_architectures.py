from __future__ import annotations

from pathlib import Path

import torch

from chess_nn_playground.ideas.implementation import validate_idea_for_training
from chess_nn_playground.models.registry import MODEL_BUILDERS
from chess_nn_playground.models.registry import available_models, build_model
from chess_nn_playground.models.research_packet_probe import ResearchPacketProbe
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES


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
}


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
        report = validate_idea_for_training(folder)
        assert report["valid"], report
