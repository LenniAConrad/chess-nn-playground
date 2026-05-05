from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.cayley_hamilton_coeffs import build_cayley_hamilton_coeffs_from_config
from chess_nn_playground.models.cayley_orthogonal import build_cayley_orthogonal_from_config
from chess_nn_playground.models.cnn import build_cnn_from_config
from chess_nn_playground.models.dykstra_lcp import build_dykstra_lcp_from_config
from chess_nn_playground.models.hadamard_spectrum import build_hadamard_spectrum_from_config
from chess_nn_playground.models.permanent_ryser import build_permanent_ryser_from_config
from chess_nn_playground.models.stable_rank_multiscale import build_stable_rank_multiscale_from_config
from chess_nn_playground.models.dykstra_vetoselect import build_dykstra_vetoselect_from_config
from chess_nn_playground.models.gpt_research_architectures import (
    build_conditional_surprisal_gate_from_config,
    build_contamination_dro_huber_tail_from_config,
    build_material_locked_tactical_dro_from_config,
    build_soft_sorting_order_ranker_from_config,
)
from chess_nn_playground.models.lc0_bt4 import build_lc0_bt4_from_config
from chess_nn_playground.models.mlp import build_mlp_from_config
from chess_nn_playground.models.nnue import build_nnue_from_config
from chess_nn_playground.models.research_architectures import (
    build_boundary_edit_from_config,
    build_chess_operator_basis_from_config,
    build_factor_agreement_from_config,
    build_neural_proof_number_from_config,
    build_null_move_contrast_from_config,
    build_obligation_flow_from_config,
    build_proof_core_from_config,
    build_response_minimax_from_config,
    build_rule_dynamics_from_config,
    build_tactical_equilibrium_from_config,
)
from chess_nn_playground.models.research_packet_probe import build_research_packet_probe_from_config
from chess_nn_playground.models.research_packet_probe import infer_mechanism_family
from chess_nn_playground.models.research_packet_registry import RESEARCH_PACKET_MODEL_NAMES
from chess_nn_playground.models.residual_cnn import build_residual_cnn_from_config
from chess_nn_playground.models.sparse_relation_pursuit import build_sparse_relation_pursuit_from_config
from chess_nn_playground.models.vetoselect import build_vetoselect_from_config


MODEL_BUILDERS = {
    "simple_cnn": build_cnn_from_config,
    "cnn_baseline": build_cnn_from_config,
    "residual_cnn": build_residual_cnn_from_config,
    "mlp": build_mlp_from_config,
    "board_mlp": build_mlp_from_config,
    "stockfish_nnue": build_nnue_from_config,
    "nnue": build_nnue_from_config,
    "lc0_bt4": build_lc0_bt4_from_config,
    "lc0_bt4_classifier": build_lc0_bt4_from_config,
    "vetoselect_positive_claim_abstention": build_vetoselect_from_config,
    "dykstra_lcp": build_dykstra_lcp_from_config,
    "dykstra_vetoselect": build_dykstra_vetoselect_from_config,
    "sparse_relation_pursuit_asymmetry": build_sparse_relation_pursuit_from_config,
    "sparse_relation_pursuit": build_sparse_relation_pursuit_from_config,
    "chess_operator_basis_classifier": build_chess_operator_basis_from_config,
    "response_minimax_classifier": build_response_minimax_from_config,
    "factor_agreement_classifier": build_factor_agreement_from_config,
    "puzzle_obligation_flow_network": build_obligation_flow_from_config,
    "null_move_contrast_puzzle_network": build_null_move_contrast_from_config,
    "proof_core_set_verifier": build_proof_core_from_config,
    "neural_proof_number_search": build_neural_proof_number_from_config,
    "boundary_edit_lagrangian_network": build_boundary_edit_from_config,
    "tactical_equilibrium_network": build_tactical_equilibrium_from_config,
    "rule_consistent_latent_dynamics": build_rule_dynamics_from_config,
    "contamination_dro_huber_tail_rejection": build_contamination_dro_huber_tail_from_config,
    "material_locked_tactical_dro": build_material_locked_tactical_dro_from_config,
    "soft_sorting_order_residual_ranker": build_soft_sorting_order_ranker_from_config,
    "conditional_surprisal_gate": build_conditional_surprisal_gate_from_config,
    "hadamard_spectrum_network": build_hadamard_spectrum_from_config,
    "cayley_orthogonal_network": build_cayley_orthogonal_from_config,
    "stable_rank_multiscale_network": build_stable_rank_multiscale_from_config,
    "permanent_ryser_network": build_permanent_ryser_from_config,
    "cayley_hamilton_coeffs_network": build_cayley_hamilton_coeffs_from_config,
}

def _make_research_packet_builder(model_name: str) -> Any:
    def build_named_research_packet(config: dict[str, Any]) -> nn.Module:
        packet_config = dict(config)
        packet_config.setdefault("name", model_name)
        packet_config.setdefault("packet_profile", model_name)
        packet_config.setdefault("mechanism_family", infer_mechanism_family(model_name))
        return build_research_packet_probe_from_config(packet_config)

    build_named_research_packet.__name__ = f"build_{model_name}_from_config"
    build_named_research_packet.__qualname__ = build_named_research_packet.__name__
    return build_named_research_packet


for _research_packet_model_name in RESEARCH_PACKET_MODEL_NAMES:
    MODEL_BUILDERS.setdefault(_research_packet_model_name, _make_research_packet_builder(_research_packet_model_name))


def register_model(name: str, builder: Any) -> None:
    if not name:
        raise ValueError("Model name must be non-empty")
    if name in MODEL_BUILDERS:
        raise ValueError(f"Model already registered: {name}")
    MODEL_BUILDERS[name] = builder


def available_models() -> list[str]:
    return sorted(MODEL_BUILDERS)


def build_model(name: str, config: dict[str, Any]) -> nn.Module:
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {name}. Available: {available_models()}")
    return MODEL_BUILDERS[name](config)
