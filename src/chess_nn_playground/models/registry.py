from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.attack_hodge_sheaf import build_attack_hodge_sheaf_from_config
from chess_nn_playground.models.cayley_hamilton_coeffs import build_cayley_hamilton_coeffs_from_config
from chess_nn_playground.models.cayley_orthogonal import build_cayley_orthogonal_from_config
from chess_nn_playground.models.cnn import build_cnn_from_config
from chess_nn_playground.models.directed_attack_sheaf import build_directed_attack_sheaf_from_config
from chess_nn_playground.models.dykstra_lcp import build_dykstra_lcp_from_config
from chess_nn_playground.models.empty_square_opportunity_network import build_empty_square_opportunity_network_from_config
from chess_nn_playground.models.hadamard_spectrum import build_hadamard_spectrum_from_config
from chess_nn_playground.models.hall_defect_obligation_matroid import (
    build_hall_defect_obligation_matroid_network_from_config,
)
from chess_nn_playground.models.geometry_pseudolikelihood_ratio import (
    build_geometry_conditioned_board_pseudo_likelihood_ratio_network_from_config,
)
from chess_nn_playground.models.global_scratchpad_boardnet import build_global_scratchpad_boardnet_from_config
from chess_nn_playground.models.blocker_pin_lattice import build_blocker_pin_lattice_network_from_config
from chess_nn_playground.models.hypercolumn_square_readout_cnn import (
    build_hypercolumn_square_readout_cnn_from_config,
)
from chess_nn_playground.models.independence_residual import (
    build_independence_residual_interaction_network_from_config,
)
from chess_nn_playground.models.king_escape_percolation import build_king_escape_percolation_network_from_config
from chess_nn_playground.models.king_shelter_microkernel import build_king_shelter_microkernel_network_from_config
from chess_nn_playground.models.latent_reply_entropy import build_latent_reply_entropy_network_from_config
from chess_nn_playground.models.mobius_piece_constellation import build_mobius_piece_constellation_network_from_config
from chess_nn_playground.models.move_landscape_net import build_move_landscape_net_from_config
from chess_nn_playground.models.multiplicative_conjunction_convnet import (
    build_multiplicative_conjunction_convnet_from_config,
)
from chess_nn_playground.models.occupancy_run_length_segment import build_occupancy_run_length_segment_encoder_from_config
from chess_nn_playground.models.oriented_tactical_sheaf import build_oriented_tactical_sheaf_from_config
from chess_nn_playground.models.permanent_ryser import build_permanent_ryser_from_config
from chess_nn_playground.models.patch_mixer_boardnet import build_patch_mixer_boardnet_from_config
from chess_nn_playground.models.piece_plane_gated_cnn import build_piece_plane_gated_cnn_from_config
from chess_nn_playground.models.safe_reply_certificate import build_safe_reply_certificate_verifier_from_config
from chess_nn_playground.models.specialist_head_cnn import build_specialist_head_cnn_from_config
from chess_nn_playground.models.sparse_witness_bottleneck import (
    build_sparse_witness_piece_bottleneck_network_from_config,
)
from chess_nn_playground.models.square_color_parity_mixer import build_square_color_parity_mixer_from_config
from chess_nn_playground.models.stable_rank_multiscale import build_stable_rank_multiscale_from_config
from chess_nn_playground.models.soft_king_cage_path import (
    build_soft_king_cage_path_bottleneck_network_from_config,
)
from chess_nn_playground.models.toda_isospectral_flow import (
    build_toda_isospectral_flow_network_from_config,
)
from chess_nn_playground.models.threat_topology_betti import (
    build_threat_topology_betti_bottleneck_network_from_config,
)
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
from chess_nn_playground.models.relational_query_algebra import (
    build_relational_query_algebra_network_from_config,
)
from chess_nn_playground.models.residual_calibration import build_residual_calibration_error_field_from_config
from chess_nn_playground.models.residual_cnn import build_residual_cnn_from_config
from chess_nn_playground.models.set_query_attention import build_set_query_attention_bottleneck_from_config
from chess_nn_playground.models.sparse_relation_pursuit import build_sparse_relation_pursuit_from_config
from chess_nn_playground.models.counterplay_insolvency_ledger import (
    build_counterplay_insolvency_ledger_from_config,
)
from chess_nn_playground.models.cross_defense_consistency import (
    build_cross_defense_consistency_network_from_config,
)
from chess_nn_playground.models.defender_opportunity_cost_auction import (
    build_defender_opportunity_cost_auction_network_from_config,
)
from chess_nn_playground.models.defender_timing_schedule import (
    build_defender_timing_schedule_network_from_config,
)
from chess_nn_playground.models.discovered_ray_switchboard import (
    build_discovered_ray_switchboard_network_from_config,
)
from chess_nn_playground.models.forced_target_funnel import (
    build_forced_target_funnel_network_from_config,
)
from chess_nn_playground.models.hierarchical_tactical_option import (
    build_hierarchical_tactical_option_network_from_config,
)
from chess_nn_playground.models.masked_codec_interaction_curvature import (
    build_masked_codec_interaction_curvature_network_from_config,
)
from chess_nn_playground.models.non_puzzle_score_curl_divergence import (
    build_non_puzzle_score_curl_divergence_bottleneck_from_config,
)
from chess_nn_playground.models.phase_specialist_calibration_mixture import (
    build_phase_specialist_calibration_mixture_from_config,
)
from chess_nn_playground.models.pinned_mobility_nullspace import (
    build_pinned_mobility_nullspace_network_from_config,
)
from chess_nn_playground.models.ray_grammar_edit_distance import (
    build_ray_grammar_edit_distance_network_from_config,
)
from chess_nn_playground.models.role_counterfactual_necessity import (
    build_role_counterfactual_necessity_network_from_config,
)
from chess_nn_playground.models.tactical_effective_resistance import (
    build_tactical_effective_resistance_network_from_config,
)
from chess_nn_playground.models.tactical_subgoal_automaton import (
    build_tactical_subgoal_automaton_network_from_config,
)
from chess_nn_playground.models.tactical_threat_sheaf import build_tactical_threat_sheaf_from_config
from chess_nn_playground.models.tensor_core_square_pair_field import (
    build_tensor_core_square_pair_field_network_from_config,
)
from chess_nn_playground.models.tiny_chess_micronet import build_tiny_chess_micronet_from_config
from chess_nn_playground.models.variational_board_action import build_variational_board_action_network_from_config
from chess_nn_playground.models.vetoselect import build_vetoselect_from_config


from chess_nn_playground.models.orbit_disagreement import build_orbit_disagreement_residual_network_from_config
from chess_nn_playground.models.hall_dual_residual import build_hall_defect_dual_residual_network_from_config
from chess_nn_playground.models.credal_temperature import build_credal_temperature_field_network_from_config
from chess_nn_playground.models.sylvester_coupling import build_sylvester_tactical_coupling_network_from_config
from chess_nn_playground.models.schur_complement_defender import build_schur_complement_defender_network_from_config
from chess_nn_playground.models.bures_wasserstein_threat import build_bures_wasserstein_threat_network_from_config
from chess_nn_playground.models.numerical_range_boundary import build_numerical_range_boundary_network_from_config
from chess_nn_playground.models.lyapunov_threat_stability import build_lyapunov_threat_stability_network_from_config
from chess_nn_playground.models.pfaffian_skew_threat import build_pfaffian_skew_threat_network_from_config
from chess_nn_playground.models.padic_ultrametric_threat import build_padic_ultrametric_threat_network_from_config
from chess_nn_playground.models.free_probability_r_transform import build_free_probability_r_transform_network_from_config


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
    "king_escape_percolation_network": build_king_escape_percolation_network_from_config,
    "soft_king_cage_path_bottleneck_network": build_soft_king_cage_path_bottleneck_network_from_config,
    "hall_defect_obligation_matroid_network": build_hall_defect_obligation_matroid_network_from_config,
    "threat_topology_betti_bottleneck_network": build_threat_topology_betti_bottleneck_network_from_config,
    "blocker_pin_lattice_network": build_blocker_pin_lattice_network_from_config,
    "safe_reply_certificate_verifier": build_safe_reply_certificate_verifier_from_config,
    "latent_reply_entropy_network": build_latent_reply_entropy_network_from_config,
    "geometry_conditioned_board_pseudo_likelihood_ratio_network": build_geometry_conditioned_board_pseudo_likelihood_ratio_network_from_config,
    "mobius_piece_constellation_network": build_mobius_piece_constellation_network_from_config,
    "sparse_witness_piece_bottleneck_network": build_sparse_witness_piece_bottleneck_network_from_config,
    "tactical_threat_sheaf_network": build_tactical_threat_sheaf_from_config,
    "oriented_tactical_sheaf_laplacian": build_oriented_tactical_sheaf_from_config,
    "attack_hodge_sheaf_tension_network": build_attack_hodge_sheaf_from_config,
    "directed_attack_sheaf_tension_network": build_directed_attack_sheaf_from_config,
    "one_ply_counterfactual_move_landscape_network": build_move_landscape_net_from_config,
    "specialist_head_cnn": build_specialist_head_cnn_from_config,
    "patch_mixer_boardnet": build_patch_mixer_boardnet_from_config,
    "piece_plane_gated_cnn": build_piece_plane_gated_cnn_from_config,
    "hypercolumn_square_readout_cnn": build_hypercolumn_square_readout_cnn_from_config,
    "multiplicative_conjunction_convnet": build_multiplicative_conjunction_convnet_from_config,
    "empty_square_opportunity_network": build_empty_square_opportunity_network_from_config,
    "global_scratchpad_boardnet": build_global_scratchpad_boardnet_from_config,
    "square_color_parity_mixer": build_square_color_parity_mixer_from_config,
    "independence_residual_interaction_network": build_independence_residual_interaction_network_from_config,
    "residual_calibration_error_field": build_residual_calibration_error_field_from_config,
    "set_query_attention_bottleneck": build_set_query_attention_bottleneck_from_config,
    "relational_query_algebra_network": build_relational_query_algebra_network_from_config,
    "toda_isospectral_flow_network": build_toda_isospectral_flow_network_from_config,
    "occupancy_run_length_segment_encoder": build_occupancy_run_length_segment_encoder_from_config,
    "king_shelter_microkernel_network": build_king_shelter_microkernel_network_from_config,
    "variational_board_action_network": build_variational_board_action_network_from_config,
    "tensor_core_square_pair_field_network": build_tensor_core_square_pair_field_network_from_config,
    "tiny_chess_micronet": build_tiny_chess_micronet_from_config,
    "hierarchical_tactical_option_network": build_hierarchical_tactical_option_network_from_config,
    "cross_defense_consistency_network": build_cross_defense_consistency_network_from_config,
    "defender_timing_schedule_network": build_defender_timing_schedule_network_from_config,
    "discovered_ray_switchboard_network": build_discovered_ray_switchboard_network_from_config,
    "counterplay_insolvency_ledger": build_counterplay_insolvency_ledger_from_config,
    "pinned_mobility_nullspace_network": build_pinned_mobility_nullspace_network_from_config,
    "tactical_effective_resistance_network": build_tactical_effective_resistance_network_from_config,
    "defender_opportunity_cost_auction_network": build_defender_opportunity_cost_auction_network_from_config,
    "role_counterfactual_necessity_network": build_role_counterfactual_necessity_network_from_config,
    "phase_specialist_calibration_mixture": build_phase_specialist_calibration_mixture_from_config,
    "forced_target_funnel_network": build_forced_target_funnel_network_from_config,
    "tactical_subgoal_automaton_network": build_tactical_subgoal_automaton_network_from_config,
    "masked_codec_interaction_curvature_network": build_masked_codec_interaction_curvature_network_from_config,
    "non_puzzle_score_curl_divergence_bottleneck": build_non_puzzle_score_curl_divergence_bottleneck_from_config,
    "ray_grammar_edit_distance_network": build_ray_grammar_edit_distance_network_from_config,
    "orbit_disagreement_residual_network": build_orbit_disagreement_residual_network_from_config,
    "hall_defect_dual_residual_network": build_hall_defect_dual_residual_network_from_config,
    "credal_temperature_field_network": build_credal_temperature_field_network_from_config,
    "sylvester_tactical_coupling_network": build_sylvester_tactical_coupling_network_from_config,
    "schur_complement_defender_network": build_schur_complement_defender_network_from_config,
    "bures_wasserstein_threat_network": build_bures_wasserstein_threat_network_from_config,
    "numerical_range_boundary_network": build_numerical_range_boundary_network_from_config,
    "lyapunov_threat_stability_network": build_lyapunov_threat_stability_network_from_config,
    "pfaffian_skew_threat_network": build_pfaffian_skew_threat_network_from_config,
    "padic_ultrametric_threat_network": build_padic_ultrametric_threat_network_from_config,
    "free_probability_r_transform_network": build_free_probability_r_transform_network_from_config,
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
