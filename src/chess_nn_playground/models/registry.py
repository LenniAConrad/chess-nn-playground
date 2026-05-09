from __future__ import annotations

from typing import Any

from torch import nn

from chess_nn_playground.models.attack_defense_sheaf import build_attack_defense_sheaf_from_config
from chess_nn_playground.models.attack_hodge_sheaf import build_attack_hodge_sheaf_from_config
from chess_nn_playground.models.cayley_hamilton_coeffs import build_cayley_hamilton_coeffs_from_config
from chess_nn_playground.models.cayley_orthogonal import build_cayley_orthogonal_from_config
from chess_nn_playground.models.centered_tempo_odd_interventional_bottleneck import (
    build_centered_tempo_odd_interventional_bottleneck_from_config,
)
from chess_nn_playground.models.chess_geometry_transport import (
    build_entropic_chess_geometry_transport_network_from_config,
)
from chess_nn_playground.models.chess_mode_tucker_relation_certificate import (
    build_chess_mode_tucker_relation_certificate_from_config,
)
from chess_nn_playground.models.cnn import build_cnn_from_config
from chess_nn_playground.models.determinantal_volume import (
    build_determinantal_tactical_volume_bottleneck_from_config,
)
from chess_nn_playground.models.differentiable_chess_fact_lattice import (
    build_differentiable_chess_fact_lattice_from_config,
)
from chess_nn_playground.models.directed_attack_sheaf import build_directed_attack_sheaf_from_config
from chess_nn_playground.models.file_mirror_tension_sheaf import build_file_mirror_tension_sheaf_from_config
from chess_nn_playground.models.finite_field_character_sum import (
    build_finite_field_character_sum_board_network_from_config,
)
from chess_nn_playground.models.schur_ray_line_algebra import (
    build_schur_ray_line_algebra_network_from_config,
)
from chess_nn_playground.models.bitboard_shift_algebra import (
    build_bitboard_shift_algebra_network_from_config,
)
from chess_nn_playground.models.dykstra_lcp import build_dykstra_lcp_from_config
from chess_nn_playground.models.auxiliary_reconstruction_boardnet import (
    build_auxiliary_reconstruction_boardnet_from_config,
)
from chess_nn_playground.models.adapter_sandwich_residual_cnn import (
    build_adapter_sandwich_residual_cnn_from_config,
)
from chess_nn_playground.models.capsule_motif_boardnet import (
    build_capsule_motif_boardnet_from_config,
)
from chess_nn_playground.models.multi_order_board_scan_network import (
    build_multi_order_board_scan_network_from_config,
)
from chess_nn_playground.models.cross_stitch_cnn_token_fusion_net import (
    build_cross_stitch_cnn_token_fusion_net_from_config,
)
from chess_nn_playground.models.neural_decision_forest_boardnet import (
    build_neural_decision_forest_boardnet_from_config,
)
from chess_nn_playground.models.vector_quantized_motif_codebook_net import (
    build_vector_quantized_motif_codebook_net_from_config,
)
from chess_nn_playground.models.agreement_variance_head_net import (
    build_agreement_variance_head_net_from_config,
)
from chess_nn_playground.models.early_exit_cascade_boardnet import (
    build_early_exit_cascade_boardnet_from_config,
)
from chess_nn_playground.models.empty_square_opportunity_network import build_empty_square_opportunity_network_from_config
from chess_nn_playground.models.entropic_piece_target_transport_bottleneck import (
    build_entropic_piece_target_transport_bottleneck_from_config,
)
from chess_nn_playground.models.piece_target_transport import (
    build_piece_target_entropic_transport_bottleneck_from_config,
)
from chess_nn_playground.models.hadamard_spectrum import build_hadamard_spectrum_from_config
from chess_nn_playground.models.harmonic_board_potential_network import (
    build_harmonic_board_potential_network_from_config,
)
from chess_nn_playground.models.tropical_constraint_circuit_network import (
    build_tropical_constraint_circuit_network_from_config,
)
from chess_nn_playground.models.grassmannian_principal_angle_bottleneck import (
    build_grassmannian_principal_angle_bottleneck_from_config,
)
from chess_nn_playground.models.matrix_pencil_generalized_spectrum_bottleneck import (
    build_matrix_pencil_generalized_spectrum_bottleneck_from_config,
)
from chess_nn_playground.models.polar_procrustes_alignment_bottleneck import (
    build_polar_procrustes_alignment_bottleneck_from_config,
)
from chess_nn_playground.models.hall_defect_obligation_matroid import (
    build_hall_defect_obligation_matroid_network_from_config,
)
from chess_nn_playground.models.hall_defect_zeta import (
    build_hall_defect_zeta_operator_from_config,
)
from chess_nn_playground.models.geometry_pseudolikelihood_ratio import (
    build_geometry_conditioned_board_pseudo_likelihood_ratio_network_from_config,
)
from chess_nn_playground.models.global_scratchpad_boardnet import build_global_scratchpad_boardnet_from_config
from chess_nn_playground.models.learnable_pooling_tree_boardnet import (
    build_learnable_pooling_tree_boardnet_from_config,
)
from chess_nn_playground.models.spatial_film_coordinate_net import (
    build_spatial_film_coordinate_net_from_config,
)
from chess_nn_playground.models.channel_bilinear_role_mixer import (
    build_channel_bilinear_role_mixer_from_config,
)
from chess_nn_playground.models.evidence_sieve_network import (
    build_evidence_sieve_network_from_config,
)
from chess_nn_playground.models.ring_shell_recurrent_boardnet import (
    build_ring_shell_recurrent_boardnet_from_config,
)
from chess_nn_playground.models.rank_file_memory_grid_net import (
    build_rank_file_memory_grid_net_from_config,
)
from chess_nn_playground.models.line_piece_crossbar_network import (
    build_line_piece_crossbar_network_from_config,
)
from chess_nn_playground.models.near_puzzle_margin_twin_network import (
    build_near_puzzle_margin_twin_network_from_config,
)
from chess_nn_playground.models.puzzle_boundary_twin_encoder import (
    build_puzzle_boundary_twin_encoder_from_config,
)
from chess_nn_playground.models.critical_square_budget_network import (
    build_critical_square_budget_network_from_config,
)
from chess_nn_playground.models.exchange_soundness_graph_network import (
    build_exchange_soundness_graph_network_from_config,
)
from chess_nn_playground.models.tactical_program_induction import (
    build_tactical_program_induction_network_from_config,
)
from chess_nn_playground.models.legal_reaction_bottleneck_network import (
    build_legal_reaction_bottleneck_network_from_config,
)
from chess_nn_playground.models.prototype_margin_puzzle_network import (
    build_prototype_margin_puzzle_network_from_config,
)
from chess_nn_playground.models.stripe_selective_mixer_cnn import (
    build_stripe_selective_mixer_cnn_from_config,
)
from chess_nn_playground.models.king_zone_evidence_ledger import (
    build_king_zone_evidence_ledger_from_config,
)
from chess_nn_playground.models.blocker_pin_lattice import build_blocker_pin_lattice_network_from_config
from chess_nn_playground.models.hypercolumn_square_readout_cnn import (
    build_hypercolumn_square_readout_cnn_from_config,
)
from chess_nn_playground.models.independence_residual import (
    build_independence_residual_interaction_network_from_config,
)
from chess_nn_playground.models.king_anchored_euler_interaction_network import (
    build_king_anchored_euler_interaction_network_from_config,
)
from chess_nn_playground.models.non_backtracking_tactical_walk import (
    build_non_backtracking_tactical_walk_network_from_config,
)
from chess_nn_playground.models.king_anchored_material_null_transport import (
    build_king_anchored_material_null_transport_bottleneck_from_config,
)
from chess_nn_playground.models.king_escape_percolation import build_king_escape_percolation_network_from_config
from chess_nn_playground.models.king_shelter_microkernel import build_king_shelter_microkernel_network_from_config
from chess_nn_playground.models.latent_reply_entropy import build_latent_reply_entropy_network_from_config
from chess_nn_playground.models.mobius_piece_constellation import build_mobius_piece_constellation_network_from_config
from chess_nn_playground.models.move_landscape_net import build_move_landscape_net_from_config
from chess_nn_playground.models.multiplicative_conjunction_convnet import (
    build_multiplicative_conjunction_convnet_from_config,
)
from chess_nn_playground.models.nuisance_orthogonal_puzzle_bottleneck import (
    build_nuisance_orthogonal_puzzle_bottleneck_from_config,
)
from chess_nn_playground.models.occupancy_run_length_segment import build_occupancy_run_length_segment_encoder_from_config
from chess_nn_playground.models.ordinal_evidence_ladder import (
    build_ordinal_evidence_ladder_network_from_config,
)
from chess_nn_playground.models.oriented_tactical_sheaf import build_oriented_tactical_sheaf_from_config
from chess_nn_playground.models.permanent_ryser import build_permanent_ryser_from_config
from chess_nn_playground.models.patch_mixer_boardnet import build_patch_mixer_boardnet_from_config
from chess_nn_playground.models.piece_plane_gated_cnn import build_piece_plane_gated_cnn_from_config
from chess_nn_playground.models.safe_reply_certificate import build_safe_reply_certificate_verifier_from_config
from chess_nn_playground.models.specialist_head_cnn import build_specialist_head_cnn_from_config
from chess_nn_playground.models.tactical_sheaf_curvature import build_tactical_sheaf_curvature_from_config
from chess_nn_playground.models.tactical_sheaf_tension import build_tactical_sheaf_tension_from_config
from chess_nn_playground.models.tactical_transport_imbalance import (
    build_tactical_transport_imbalance_network_from_config,
)
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
from chess_nn_playground.models.counterfactual_defender_dropout import (
    build_counterfactual_defender_dropout_network_from_config,
)
from chess_nn_playground.models.exchange_then_king_dual_stream import (
    build_exchange_then_king_dual_stream_from_config,
)
from chess_nn_playground.models.source_invariant_puzzle_bottleneck import (
    build_source_invariant_puzzle_bottleneck_from_config,
)
from chess_nn_playground.models.reply_set_contrastive_transformer import (
    build_reply_set_contrastive_transformer_from_config,
)
from chess_nn_playground.models.tactical_symptom_bayesian_network import (
    build_tactical_symptom_bayesian_network_from_config,
)
from chess_nn_playground.models.counterfactual_delta_bottleneck import (
    build_counterfactual_delta_bottleneck_from_config,
)
from chess_nn_playground.models.counterfactual_move_delta_spectrum import (
    build_counterfactual_move_delta_spectrum_network_from_config,
)
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
from chess_nn_playground.models.masked_surprise_codec import (
    build_masked_board_code_length_surprise_network_from_config,
)
from chess_nn_playground.models.non_puzzle_score_curl_divergence import (
    build_non_puzzle_score_curl_divergence_bottleneck_from_config,
)
from chess_nn_playground.models.non_puzzle_score_field_bottleneck import (
    build_non_puzzle_score_field_bottleneck_network_from_config,
)
from chess_nn_playground.models.soft_formal_concept_closure import (
    build_soft_formal_concept_closure_network_from_config,
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
from chess_nn_playground.models.kinematic_commutator_bottleneck import (
    build_kinematic_commutator_bottleneck_network_from_config,
)
from chess_nn_playground.models.legal_automorphism_quotient_network import (
    build_legal_automorphism_quotient_network_from_config,
)
from chess_nn_playground.models.color_flip_orbit_evidence import (
    build_color_flip_orbit_evidence_bottleneck_from_config,
)
from chess_nn_playground.models.rule_automorphism_quotient import (
    build_rule_automorphism_quotient_bottleneck_from_config,
)
from chess_nn_playground.models.rule_exact_orbit_bottleneck import (
    build_rule_exact_orbit_bottleneck_from_config,
)
from chess_nn_playground.models.rule_partition_invariant_bottleneck import (
    build_side_canonical_rule_partition_invariant_bottleneck_from_config,
)
from chess_nn_playground.models.ray_language_automaton_network import (
    build_ray_language_automaton_network_from_config,
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
from chess_nn_playground.models.tempo_odd_bottleneck import build_tempo_odd_bottleneck_from_config
from chess_nn_playground.models.tensor_core_square_pair_field import (
    build_tensor_core_square_pair_field_network_from_config,
)
from chess_nn_playground.models.tiny_chess_micronet import build_tiny_chess_micronet_from_config
from chess_nn_playground.models.variational_board_action import build_variational_board_action_network_from_config
from chess_nn_playground.models.vetoselect import build_vetoselect_from_config


from chess_nn_playground.models.orbit_disagreement import build_orbit_disagreement_residual_network_from_config
from chess_nn_playground.models.attention_disagreement_residual_network import (
    build_attention_disagreement_residual_network_from_config,
)
from chess_nn_playground.models.cross_scale_attention_residual_network import (
    build_cross_scale_attention_residual_network_from_config,
)
from chess_nn_playground.models.slot_attention_role_binding_network import (
    build_slot_attention_role_binding_network_from_config,
)
from chess_nn_playground.models.attention_perturbation_sensitivity_network import (
    build_attention_perturbation_sensitivity_network_from_config,
)
from chess_nn_playground.models.kernel_mean_prototype_network import (
    build_kernel_mean_prototype_network_from_config,
)
from chess_nn_playground.models.tensorsketch_interaction_network import (
    build_tensorsketch_interaction_network_from_config,
)
from chess_nn_playground.models.maxout_region_signature_network import (
    build_maxout_region_signature_network_from_config,
)
from chess_nn_playground.models.spline_board_surface_network import (
    build_spline_board_surface_network_from_config,
)
from chess_nn_playground.models.boundary_condition_disagreement_cnn import (
    build_boundary_condition_disagreement_cnn_from_config,
)
from chess_nn_playground.models.piece_drop_stability_network import (
    build_piece_drop_stability_network_from_config,
)
from chess_nn_playground.models.hall_dual_residual import build_hall_defect_dual_residual_network_from_config
from chess_nn_playground.models.credal_near_puzzle_evidence import (
    build_credal_near_puzzle_evidence_network_from_config,
)
from chess_nn_playground.models.credal_temperature import build_credal_temperature_field_network_from_config
from chess_nn_playground.models.sylvester_coupling import build_sylvester_tactical_coupling_network_from_config
from chess_nn_playground.models.schur_complement_defender import build_schur_complement_defender_network_from_config
from chess_nn_playground.models.bures_wasserstein_threat import build_bures_wasserstein_threat_network_from_config
from chess_nn_playground.models.numerical_range_boundary import build_numerical_range_boundary_network_from_config
from chess_nn_playground.models.lyapunov_threat_stability import build_lyapunov_threat_stability_network_from_config
from chess_nn_playground.models.pfaffian_skew_threat import build_pfaffian_skew_threat_network_from_config
from chess_nn_playground.models.padic_ultrametric_threat import build_padic_ultrametric_threat_network_from_config
from chess_nn_playground.models.free_probability_r_transform import build_free_probability_r_transform_network_from_config
from chess_nn_playground.models.williamson_symplectic_threat_network import (
    build_williamson_symplectic_threat_network_from_config,
)
from chess_nn_playground.models.magnus_bch_coupling_series_network import (
    build_magnus_bch_coupling_series_network_from_config,
)
from chess_nn_playground.models.riccati_optimal_defense_network import (
    build_riccati_optimal_defense_network_from_config,
)
from chess_nn_playground.models.clifford_rotor_threat_network import (
    build_clifford_rotor_threat_network_from_config,
)
from chess_nn_playground.models.tracy_widom_level_spacing_network import (
    build_tracy_widom_level_spacing_network_from_config,
)
from chess_nn_playground.models.lindstrom_gessel_viennot_path_network import (
    build_lindstrom_gessel_viennot_path_network_from_config,
)
from chess_nn_playground.models.local_neighborhood_geometry_network import (
    build_local_neighborhood_geometry_network_from_config,
)
from chess_nn_playground.models.multi_scale_dilated_board_mixer_cnn import (
    build_multi_scale_dilated_board_mixer_cnn_from_config,
)
from chess_nn_playground.models.piece_token_cnn_hybrid import (
    build_piece_token_cnn_hybrid_from_config,
)
from chess_nn_playground.models.puzzle_binary_benchmark_challengers import (
    build_negative_class_disentangled_puzzle_head_from_config,
    build_puzzle_binary_benchmark_challengers_from_config,
)
from chess_nn_playground.models.tactical_bisimulation_puzzle_network import (
    build_tactical_bisimulation_puzzle_network_from_config,
)
from chess_nn_playground.models.krylov_tactical_subspace_network import (
    build_krylov_tactical_subspace_network_from_config,
)
from chess_nn_playground.models.adaptive_tactical_resolvent_network import (
    build_adaptive_tactical_resolvent_network_from_config,
)
from chess_nn_playground.models.tactical_controllability_gramian_network import (
    build_tactical_controllability_gramian_network_from_config,
)
from chess_nn_playground.models.support_polar_zonotope import (
    build_support_polar_zonotope_certificate_network_from_config,
)
from chess_nn_playground.models.loop_frustration_curvature_network import (
    build_loop_frustration_curvature_network_from_config,
)
from chess_nn_playground.models.forcing_response_front_door_bottleneck import (
    build_forcing_response_front_door_bottleneck_from_config,
)
from chess_nn_playground.models.causal_piece_derivative_network import (
    build_causal_piece_derivative_network_from_config,
)
from chess_nn_playground.models.phase_transition_pressure_network import (
    build_phase_transition_pressure_network_from_config,
)
from chess_nn_playground.models.disproof_ledger_puzzle_network import (
    build_disproof_ledger_puzzle_network_from_config,
)
from chess_nn_playground.models.motif_tensor_factorization_network import (
    build_motif_tensor_factorization_network_from_config,
)
from chess_nn_playground.models.tempo_alignment_gate_network import (
    build_tempo_alignment_gate_network_from_config,
)
from chess_nn_playground.models.forcing_certificate_transformer import (
    build_forcing_certificate_transformer_from_config,
)
from chess_nn_playground.models.chess_hypercut_polynomial import (
    build_chess_hypercut_polynomial_network_from_config,
)
from chess_nn_playground.models.fisher_geodesic_tension import (
    build_fisher_geodesic_tension_network_from_config,
)
from chess_nn_playground.models.typed_hypergraph_motif_grammar import (
    build_typed_hypergraph_motif_grammar_from_config,
)
from chess_nn_playground.models.tactical_radius_filtration import (
    build_tactical_radius_filtration_from_config,
)
from chess_nn_playground.models.tactical_state_bottleneck import (
    build_tactical_state_bottleneck_from_config,
)
from chess_nn_playground.models.bounded_board_hinge_logic import (
    build_bounded_board_hinge_logic_from_config,
)
from chess_nn_playground.models.traced_threat_motif import (
    build_traced_threat_motif_network_from_config,
)
from chess_nn_playground.models.parity_syndrome import (
    build_parity_syndrome_puzzle_bottleneck_from_config,
)
from chess_nn_playground.models.wavelet_scattering_board_network import (
    build_wavelet_scattering_board_network_from_config,
)
from chess_nn_playground.models.convex_feasibility import (
    build_convex_feasibility_residual_network_from_config,
)
from chess_nn_playground.models.oriented_matroid_covector import (
    build_oriented_matroid_covector_bottleneck_from_config,
)
from chess_nn_playground.models.fixed_point_residual import (
    build_fixed_point_residual_defect_network_from_config,
)
from chess_nn_playground.models.baseline_logit_residual_adapter import (
    build_baseline_logit_residual_adapter_from_config,
)
from chess_nn_playground.models.coarse_to_fine_residual_pyramid import (
    build_coarse_to_fine_board_residual_pyramid_from_config,
)
from chess_nn_playground.models.row_file_factor_mixer import (
    build_row_file_factor_mixer_from_config,
)
from chess_nn_playground.models.piece_conditioned_hypernetwork_cnn import (
    build_piece_conditioned_hypernetwork_cnn_from_config,
)
from chess_nn_playground.models.neural_board_cellular_automaton import (
    build_neural_board_cellular_automaton_from_config,
)
from chess_nn_playground.models.symmetric_difference_twin_encoder import (
    build_symmetric_difference_twin_encoder_from_config,
)
from chess_nn_playground.models.minimal_edit_puzzle_distance_network import (
    build_minimal_edit_puzzle_distance_network_from_config,
)
from chess_nn_playground.models.barrier_cut_puzzle_network import (
    build_barrier_cut_puzzle_network_from_config,
)
from chess_nn_playground.models.tactical_hessian_spectrum_network import (
    build_tactical_hessian_spectrum_network_from_config,
)
from chess_nn_playground.models.absorbing_threat_markov_network import (
    build_absorbing_threat_markov_network_from_config,
)
from chess_nn_playground.models.neural_clause_resolution_puzzle_network import (
    build_neural_clause_resolution_puzzle_network_from_config,
)
from chess_nn_playground.models.piece_liability_gradient_network import (
    build_piece_liability_gradient_network_from_config,
)
from chess_nn_playground.models.prototype_patch_dictionary_network import (
    build_prototype_patch_dictionary_network_from_config,
)
from chess_nn_playground.models.tensor_ring_square_interaction_network import (
    build_tensor_ring_square_interaction_network_from_config,
)
from chess_nn_playground.models.sinkhorn_role_assignment_network import (
    build_sinkhorn_role_assignment_network_from_config,
)
from chess_nn_playground.models.morphological_threat_field_network import (
    build_morphological_threat_field_network_from_config,
)
from chess_nn_playground.models.invertible_board_coupling_network import (
    build_invertible_board_coupling_network_from_config,
)
from chess_nn_playground.models.sparse_expert_board_router import (
    build_sparse_expert_board_router_from_config,
)
from chess_nn_playground.models.rank_quantile import (
    build_rank_quantile_evidence_field_network_from_config,
)
from chess_nn_playground.models.ray_state_space_scan import (
    build_ray_state_space_scan_network_from_config,
)
from chess_nn_playground.models.pawn_skeleton_barrier import (
    build_pawn_skeleton_barrier_network_from_config,
)
from chess_nn_playground.models.material_phase_low_rank_adapter import (
    build_material_phase_low_rank_adapter_network_from_config,
)
from chess_nn_playground.models.replicator_payoff_piece_dynamics import (
    build_replicator_payoff_piece_dynamics_from_config,
)
from chess_nn_playground.models.differentiable_bitboard_boolean_network import (
    build_differentiable_bitboard_boolean_network_from_config,
)
from chess_nn_playground.models.orthogonal_board_moment_network import (
    build_orthogonal_board_moment_network_from_config,
)
from chess_nn_playground.models.legal_constraint_projection_residual_network import (
    build_legal_constraint_projection_residual_network_from_config,
)
from chess_nn_playground.models.zobrist_kernel_feature_network import (
    build_zobrist_kernel_feature_network_from_config,
)
from chess_nn_playground.models.low_rank_signed_cut_query_network import (
    build_low_rank_signed_cut_query_network_from_config,
)
from chess_nn_playground.models.soft_majorization_line_sorter import (
    build_soft_majorization_line_sorter_from_config,
)
from chess_nn_playground.models.convnext_boardnet import (
    build_convnext_boardnet_from_config,
)
from chess_nn_playground.models.iterative_logit_refinement_cnn import (
    build_iterative_logit_refinement_cnn_from_config,
)


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
    "hall_defect_zeta_operator": build_hall_defect_zeta_operator_from_config,
    "threat_topology_betti_bottleneck_network": build_threat_topology_betti_bottleneck_network_from_config,
    "blocker_pin_lattice_network": build_blocker_pin_lattice_network_from_config,
    "safe_reply_certificate_verifier": build_safe_reply_certificate_verifier_from_config,
    "latent_reply_entropy_network": build_latent_reply_entropy_network_from_config,
    "geometry_conditioned_board_pseudo_likelihood_ratio_network": build_geometry_conditioned_board_pseudo_likelihood_ratio_network_from_config,
    "mobius_piece_constellation_network": build_mobius_piece_constellation_network_from_config,
    "sparse_witness_piece_bottleneck_network": build_sparse_witness_piece_bottleneck_network_from_config,
    "tactical_threat_sheaf_network": build_tactical_threat_sheaf_from_config,
    "oriented_tactical_sheaf_laplacian": build_oriented_tactical_sheaf_from_config,
    "tactical_sheaf_curvature_network": build_tactical_sheaf_curvature_from_config,
    "tactical_sheaf_tension_network": build_tactical_sheaf_tension_from_config,
    "attack_defense_sheaf_energy_network": build_attack_defense_sheaf_from_config,
    "attack_hodge_sheaf_tension_network": build_attack_hodge_sheaf_from_config,
    "directed_attack_sheaf_tension_network": build_directed_attack_sheaf_from_config,
    "file_mirror_tension_sheaf": build_file_mirror_tension_sheaf_from_config,
    "entropic_piece_target_transport_bottleneck": build_entropic_piece_target_transport_bottleneck_from_config,
    "piece_target_entropic_transport_bottleneck": build_piece_target_entropic_transport_bottleneck_from_config,
    "king_anchored_material_null_transport_bottleneck": build_king_anchored_material_null_transport_bottleneck_from_config,
    "nuisance_orthogonal_puzzle_bottleneck": build_nuisance_orthogonal_puzzle_bottleneck_from_config,
    "one_ply_counterfactual_move_landscape_network": build_move_landscape_net_from_config,
    "counterfactual_move_delta_spectrum_network": build_counterfactual_move_delta_spectrum_network_from_config,
    "rule_only_counterfactual_move_delta_bottleneck": build_counterfactual_delta_bottleneck_from_config,
    "specialist_head_cnn": build_specialist_head_cnn_from_config,
    "patch_mixer_boardnet": build_patch_mixer_boardnet_from_config,
    "piece_plane_gated_cnn": build_piece_plane_gated_cnn_from_config,
    "hypercolumn_square_readout_cnn": build_hypercolumn_square_readout_cnn_from_config,
    "multiplicative_conjunction_convnet": build_multiplicative_conjunction_convnet_from_config,
    "empty_square_opportunity_network": build_empty_square_opportunity_network_from_config,
    "global_scratchpad_boardnet": build_global_scratchpad_boardnet_from_config,
    "learnable_pooling_tree_boardnet": build_learnable_pooling_tree_boardnet_from_config,
    "spatial_film_coordinate_net": build_spatial_film_coordinate_net_from_config,
    "channel_bilinear_role_mixer": build_channel_bilinear_role_mixer_from_config,
    "evidence_sieve_network": build_evidence_sieve_network_from_config,
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
    "masked_board_code_length_surprise_network": build_masked_board_code_length_surprise_network_from_config,
    "non_puzzle_score_curl_divergence_bottleneck": build_non_puzzle_score_curl_divergence_bottleneck_from_config,
    "non_puzzle_score_field_bottleneck_network": build_non_puzzle_score_field_bottleneck_network_from_config,
    "soft_formal_concept_closure_network": build_soft_formal_concept_closure_network_from_config,
    "ray_grammar_edit_distance_network": build_ray_grammar_edit_distance_network_from_config,
    "ray_language_automaton_network": build_ray_language_automaton_network_from_config,
    "kinematic_commutator_bottleneck_network": build_kinematic_commutator_bottleneck_network_from_config,
    "orbit_disagreement_residual_network": build_orbit_disagreement_residual_network_from_config,
    "hall_defect_dual_residual_network": build_hall_defect_dual_residual_network_from_config,
    "credal_near_puzzle_evidence_network": build_credal_near_puzzle_evidence_network_from_config,
    "credal_temperature_field_network": build_credal_temperature_field_network_from_config,
    "sylvester_tactical_coupling_network": build_sylvester_tactical_coupling_network_from_config,
    "schur_complement_defender_network": build_schur_complement_defender_network_from_config,
    "bures_wasserstein_threat_network": build_bures_wasserstein_threat_network_from_config,
    "numerical_range_boundary_network": build_numerical_range_boundary_network_from_config,
    "lyapunov_threat_stability_network": build_lyapunov_threat_stability_network_from_config,
    "pfaffian_skew_threat_network": build_pfaffian_skew_threat_network_from_config,
    "padic_ultrametric_threat_network": build_padic_ultrametric_threat_network_from_config,
    "free_probability_r_transform_network": build_free_probability_r_transform_network_from_config,
    "williamson_symplectic_threat_network": build_williamson_symplectic_threat_network_from_config,
    "magnus_bch_coupling_series_network": build_magnus_bch_coupling_series_network_from_config,
    "riccati_optimal_defense_network": build_riccati_optimal_defense_network_from_config,
    "clifford_rotor_threat_network": build_clifford_rotor_threat_network_from_config,
    "tracy_widom_level_spacing_network": build_tracy_widom_level_spacing_network_from_config,
    "lindstrom_gessel_viennot_path_network": build_lindstrom_gessel_viennot_path_network_from_config,
    "local_neighborhood_geometry_network": build_local_neighborhood_geometry_network_from_config,
    "tactical_transport_imbalance_network": build_tactical_transport_imbalance_network_from_config,
    "entropic_chess_geometry_transport_network": build_entropic_chess_geometry_transport_network_from_config,
    "ordinal_evidence_ladder_network": build_ordinal_evidence_ladder_network_from_config,
    "centered_tempo_odd_interventional_bottleneck": build_centered_tempo_odd_interventional_bottleneck_from_config,
    "legal_automorphism_quotient_network": build_legal_automorphism_quotient_network_from_config,
    "side_canonical_rule_partition_invariant_bottleneck": build_side_canonical_rule_partition_invariant_bottleneck_from_config,
    "rule_exact_orbit_bottleneck_network": build_rule_exact_orbit_bottleneck_from_config,
    "color_flip_orbit_evidence_bottleneck": build_color_flip_orbit_evidence_bottleneck_from_config,
    "rule_automorphism_quotient_bottleneck_network": build_rule_automorphism_quotient_bottleneck_from_config,
    "tempo_odd_bottleneck_network": build_tempo_odd_bottleneck_from_config,
    "king_anchored_euler_interaction_network": build_king_anchored_euler_interaction_network_from_config,
    "non_backtracking_tactical_walk_network": build_non_backtracking_tactical_walk_network_from_config,
    "determinantal_tactical_volume_bottleneck": build_determinantal_tactical_volume_bottleneck_from_config,
    "harmonic_board_potential_network": build_harmonic_board_potential_network_from_config,
    "tropical_constraint_circuit_network": build_tropical_constraint_circuit_network_from_config,
    "grassmannian_principal_angle_bottleneck": build_grassmannian_principal_angle_bottleneck_from_config,
    "matrix_pencil_generalized_spectrum_bottleneck": build_matrix_pencil_generalized_spectrum_bottleneck_from_config,
    "polar_procrustes_alignment_bottleneck": build_polar_procrustes_alignment_bottleneck_from_config,
    "multi_scale_dilated_board_mixer_cnn": build_multi_scale_dilated_board_mixer_cnn_from_config,
    "piece_token_cnn_hybrid": build_piece_token_cnn_hybrid_from_config,
    "finite_field_character_sum_board_network": build_finite_field_character_sum_board_network_from_config,
    "schur_ray_line_algebra_network": build_schur_ray_line_algebra_network_from_config,
    "bitboard_shift_algebra_network": build_bitboard_shift_algebra_network_from_config,
    "puzzle_binary_benchmark_challengers": build_puzzle_binary_benchmark_challengers_from_config,
    "negative_class_disentangled_puzzle_head": build_negative_class_disentangled_puzzle_head_from_config,
    "tactical_bisimulation_puzzle_network": build_tactical_bisimulation_puzzle_network_from_config,
    "krylov_tactical_subspace_network": build_krylov_tactical_subspace_network_from_config,
    "adaptive_tactical_resolvent_network": build_adaptive_tactical_resolvent_network_from_config,
    "tactical_controllability_gramian_network": build_tactical_controllability_gramian_network_from_config,
    "support_polar_zonotope_certificate_network": build_support_polar_zonotope_certificate_network_from_config,
    "loop_frustration_curvature_network": build_loop_frustration_curvature_network_from_config,
    "forcing_response_front_door_bottleneck": build_forcing_response_front_door_bottleneck_from_config,
    "chess_hypercut_polynomial_network": build_chess_hypercut_polynomial_network_from_config,
    "fisher_geodesic_tension_network": build_fisher_geodesic_tension_network_from_config,
    "typed_hypergraph_motif_grammar": build_typed_hypergraph_motif_grammar_from_config,
    "differentiable_chess_fact_lattice": build_differentiable_chess_fact_lattice_from_config,
    "tactical_radius_filtration": build_tactical_radius_filtration_from_config,
    "chess_mode_tucker_relation_certificate": build_chess_mode_tucker_relation_certificate_from_config,
    "tactical_state_bottleneck_inference": build_tactical_state_bottleneck_from_config,
    "traced_threat_motif_network": build_traced_threat_motif_network_from_config,
    "bounded_board_hinge_logic": build_bounded_board_hinge_logic_from_config,
    "parity_syndrome_puzzle_bottleneck": build_parity_syndrome_puzzle_bottleneck_from_config,
    "wavelet_scattering_board_network": build_wavelet_scattering_board_network_from_config,
    "convex_feasibility_residual_network": build_convex_feasibility_residual_network_from_config,
    "rank_quantile_evidence_field_network": build_rank_quantile_evidence_field_network_from_config,
    "oriented_matroid_covector_bottleneck": build_oriented_matroid_covector_bottleneck_from_config,
    "fixed_point_residual_defect_network": build_fixed_point_residual_defect_network_from_config,
    "baseline_logit_residual_adapter": build_baseline_logit_residual_adapter_from_config,
    "coarse_to_fine_board_residual_pyramid": build_coarse_to_fine_board_residual_pyramid_from_config,
    "attention_disagreement_residual_network": build_attention_disagreement_residual_network_from_config,
    "cross_scale_attention_residual_network": build_cross_scale_attention_residual_network_from_config,
    "slot_attention_role_binding_network": build_slot_attention_role_binding_network_from_config,
    "attention_perturbation_sensitivity_network": build_attention_perturbation_sensitivity_network_from_config,
    "kernel_mean_prototype_network": build_kernel_mean_prototype_network_from_config,
    "tensorsketch_interaction_network": build_tensorsketch_interaction_network_from_config,
    "maxout_region_signature_network": build_maxout_region_signature_network_from_config,
    "spline_board_surface_network": build_spline_board_surface_network_from_config,
    "boundary_condition_disagreement_cnn": build_boundary_condition_disagreement_cnn_from_config,
    "piece_drop_stability_network": build_piece_drop_stability_network_from_config,
    "row_file_factor_mixer": build_row_file_factor_mixer_from_config,
    "piece_conditioned_hypernetwork_cnn": build_piece_conditioned_hypernetwork_cnn_from_config,
    "neural_board_cellular_automaton": build_neural_board_cellular_automaton_from_config,
    "symmetric_difference_twin_encoder": build_symmetric_difference_twin_encoder_from_config,
    "minimal_edit_puzzle_distance_network": build_minimal_edit_puzzle_distance_network_from_config,
    "barrier_cut_puzzle_network": build_barrier_cut_puzzle_network_from_config,
    "tactical_hessian_spectrum_network": build_tactical_hessian_spectrum_network_from_config,
    "absorbing_threat_markov_network": build_absorbing_threat_markov_network_from_config,
    "neural_clause_resolution_puzzle_network": build_neural_clause_resolution_puzzle_network_from_config,
    "piece_liability_gradient_network": build_piece_liability_gradient_network_from_config,
    "prototype_patch_dictionary_network": build_prototype_patch_dictionary_network_from_config,
    "tensor_ring_square_interaction_network": build_tensor_ring_square_interaction_network_from_config,
    "sinkhorn_role_assignment_network": build_sinkhorn_role_assignment_network_from_config,
    "morphological_threat_field_network": build_morphological_threat_field_network_from_config,
    "invertible_board_coupling_network": build_invertible_board_coupling_network_from_config,
    "sparse_expert_board_router": build_sparse_expert_board_router_from_config,
    "ray_state_space_scan_network": build_ray_state_space_scan_network_from_config,
    "pawn_skeleton_barrier_network": build_pawn_skeleton_barrier_network_from_config,
    "material_phase_low_rank_adapter_network": build_material_phase_low_rank_adapter_network_from_config,
    "replicator_payoff_piece_dynamics": build_replicator_payoff_piece_dynamics_from_config,
    "differentiable_bitboard_boolean_network": build_differentiable_bitboard_boolean_network_from_config,
    "orthogonal_board_moment_network": build_orthogonal_board_moment_network_from_config,
    "legal_constraint_projection_residual_network": build_legal_constraint_projection_residual_network_from_config,
    "zobrist_kernel_feature_network": build_zobrist_kernel_feature_network_from_config,
    "low_rank_signed_cut_query_network": build_low_rank_signed_cut_query_network_from_config,
    "soft_majorization_line_sorter": build_soft_majorization_line_sorter_from_config,
    "convnext_boardnet": build_convnext_boardnet_from_config,
    "iterative_logit_refinement_cnn": build_iterative_logit_refinement_cnn_from_config,
    "early_exit_cascade_boardnet": build_early_exit_cascade_boardnet_from_config,
    "auxiliary_reconstruction_boardnet": build_auxiliary_reconstruction_boardnet_from_config,
    "agreement_variance_head_net": build_agreement_variance_head_net_from_config,
    "adapter_sandwich_residual_cnn": build_adapter_sandwich_residual_cnn_from_config,
    "capsule_motif_boardnet": build_capsule_motif_boardnet_from_config,
    "multi_order_board_scan_network": build_multi_order_board_scan_network_from_config,
    "cross_stitch_cnn_token_fusion_net": build_cross_stitch_cnn_token_fusion_net_from_config,
    "neural_decision_forest_boardnet": build_neural_decision_forest_boardnet_from_config,
    "vector_quantized_motif_codebook_net": build_vector_quantized_motif_codebook_net_from_config,
    "ring_shell_recurrent_boardnet": build_ring_shell_recurrent_boardnet_from_config,
    "rank_file_memory_grid_net": build_rank_file_memory_grid_net_from_config,
    "line_piece_crossbar_network": build_line_piece_crossbar_network_from_config,
    "near_puzzle_margin_twin_network": build_near_puzzle_margin_twin_network_from_config,
    "puzzle_boundary_twin_encoder": build_puzzle_boundary_twin_encoder_from_config,
    "critical_square_budget_network": build_critical_square_budget_network_from_config,
    "legal_reaction_bottleneck_network": build_legal_reaction_bottleneck_network_from_config,
    "exchange_soundness_graph_network": build_exchange_soundness_graph_network_from_config,
    "tactical_program_induction_network": build_tactical_program_induction_network_from_config,
    "prototype_margin_puzzle_network": build_prototype_margin_puzzle_network_from_config,
    "stripe_selective_mixer_cnn": build_stripe_selective_mixer_cnn_from_config,
    "king_zone_evidence_ledger": build_king_zone_evidence_ledger_from_config,
    "forcing_certificate_transformer": build_forcing_certificate_transformer_from_config,
    "causal_piece_derivative_network": build_causal_piece_derivative_network_from_config,
    "phase_transition_pressure_network": build_phase_transition_pressure_network_from_config,
    "disproof_ledger_puzzle_network": build_disproof_ledger_puzzle_network_from_config,
    "motif_tensor_factorization_network": build_motif_tensor_factorization_network_from_config,
    "tempo_alignment_gate_network": build_tempo_alignment_gate_network_from_config,
    "counterfactual_defender_dropout_network": build_counterfactual_defender_dropout_network_from_config,
    "exchange_then_king_dual_stream": build_exchange_then_king_dual_stream_from_config,
    "tactical_symptom_bayesian_network": build_tactical_symptom_bayesian_network_from_config,
    "source_invariant_puzzle_bottleneck": build_source_invariant_puzzle_bottleneck_from_config,
    "reply_set_contrastive_transformer": build_reply_set_contrastive_transformer_from_config,
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
