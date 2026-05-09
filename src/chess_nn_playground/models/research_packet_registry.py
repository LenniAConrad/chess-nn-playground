from __future__ import annotations


RESEARCH_PACKET_MODEL_NAMES: list[str] = ['oriented_tactical_sheaf_laplacian', 'tactical_sheaf_curvature_network', 'attack_defense_sheaf_energy_network', 'tactical_sheaf_tension_network', 'tactical_threat_sheaf_network', 'attack_hodge_sheaf_tension_network', 'directed_attack_sheaf_tension_network', 'one_ply_counterfactual_move_landscape_network', 'counterfactual_move_delta_spectrum_network', 'rule_only_counterfactual_move_delta_bottleneck', 'file_mirror_tension_sheaf', 'entropic_piece_target_transport_bottleneck', 'nuisance_orthogonal_puzzle_bottleneck', 'tactical_transport_imbalance_network', 'king_anchored_material_null_transport_bottleneck', 'piece_target_entropic_transport_bottleneck', 'entropic_chess_geometry_transport_network', 'ordinal_evidence_ladder_network', 'geometry_conditioned_board_pseudo_likelihood_ratio_network', 'mobius_piece_constellation_network', 'sparse_witness_piece_bottleneck_network', 'ray_language_automaton_network', 'kinematic_commutator_bottleneck_network', 'centered_tempo_odd_interventional_bottleneck', 'legal_automorphism_quotient_network', 'side_canonical_rule_partition_invariant_bottleneck', 'masked_board_code_length_surprise_network', 'credal_near_puzzle_evidence_network', 'rule_exact_orbit_bottleneck_network', 'color_flip_orbit_evidence_bottleneck', 'rule_automorphism_quotient_bottleneck_network', 'tempo_odd_bottleneck_network', 'king_anchored_euler_interaction_network', 'king_escape_percolation_network', 'soft_king_cage_path_bottleneck_network', 'hall_defect_obligation_matroid_network', 'threat_topology_betti_bottleneck_network', 'non_backtracking_tactical_walk_network', 'non_puzzle_score_field_bottleneck_network', 'soft_formal_concept_closure_network', 'determinantal_tactical_volume_bottleneck', 'harmonic_board_potential_network', 'tropical_constraint_circuit_network', 'grassmannian_principal_angle_bottleneck', 'matrix_pencil_generalized_spectrum_bottleneck', 'polar_procrustes_alignment_bottleneck', 'multi_scale_dilated_board_mixer_cnn', 'piece_token_cnn_hybrid', 'bispectral_phase_coupling_board_network', 'finite_field_character_sum_board_network', 'schur_ray_line_algebra_network', 'bitboard_shift_algebra_network', 'relational_query_algebra_network', 'variational_board_action_network', 'tensor_core_square_pair_field_network', 'tiny_chess_micronet', 'puzzle_binary_benchmark_challengers', 'tactical_bisimulation_puzzle_network', 'krylov_tactical_subspace_network', 'adaptive_tactical_resolvent_network', 'tactical_controllability_gramian_network', 'support_polar_zonotope_certificate_network', 'loop_frustration_curvature_network', 'forcing_response_front_door_bottleneck', 'chess_hypercut_polynomial_network', 'fisher_geodesic_tension_network', 'typed_hypergraph_motif_grammar', 'hall_defect_zeta_operator', 'differentiable_chess_fact_lattice', 'tactical_radius_filtration', 'traced_threat_motif_network', 'bounded_board_hinge_logic', 'chess_mode_tucker_relation_certificate', 'tactical_state_bottleneck_inference', 'parity_syndrome_puzzle_bottleneck', 'wavelet_scattering_board_network', 'convex_feasibility_residual_network', 'rank_quantile_evidence_field_network', 'oriented_matroid_covector_bottleneck', 'fixed_point_residual_defect_network', 'baseline_logit_residual_adapter', 'coarse_to_fine_board_residual_pyramid', 'independence_residual_interaction_network', 'residual_calibration_error_field', 'set_query_attention_bottleneck', 'attention_disagreement_residual_network', 'cross_scale_attention_residual_network', 'slot_attention_role_binding_network', 'attention_perturbation_sensitivity_network', 'kernel_mean_prototype_network', 'tensorsketch_interaction_network', 'maxout_region_signature_network', 'spline_board_surface_network', 'boundary_condition_disagreement_cnn', 'piece_drop_stability_network', 'row_file_factor_mixer', 'piece_conditioned_hypernetwork_cnn', 'neural_board_cellular_automaton', 'symmetric_difference_twin_encoder', 'prototype_patch_dictionary_network', 'channel_dropout_consensus_network', 'tensor_ring_square_interaction_network', 'sinkhorn_role_assignment_network', 'morphological_threat_field_network', 'invertible_board_coupling_network', 'sparse_expert_board_router', 'local_neighborhood_geometry_network', 'ray_state_space_scan_network', 'pawn_skeleton_barrier_network', 'square_color_parity_mixer', 'occupancy_run_length_segment_encoder', 'king_shelter_microkernel_network', 'material_phase_low_rank_adapter_network', 'replicator_payoff_piece_dynamics', 'differentiable_bitboard_boolean_network', 'orthogonal_board_moment_network', 'legal_constraint_projection_residual_network', 'zobrist_kernel_feature_network', 'low_rank_signed_cut_query_network', 'commutative_view_consistency_network', 'support_function_envelope_network', 'soft_majorization_line_sorter', 'low_displacement_rank_board_operator', 'submodular_coverage_bottleneck', 'pivot_trace_elimination_network', 'convnext_boardnet', 'board_fpn_cnn', 'piece_plane_gated_cnn', 'patch_mixer_boardnet', 'specialist_head_cnn', 'shallow_wide_residual_boardnet', 'axial_rank_file_convnet', 'early_exit_cascade_boardnet', 'auxiliary_reconstruction_boardnet', 'iterative_logit_refinement_cnn', 'agreement_variance_head_net', 'adapter_sandwich_residual_cnn', 'capsule_motif_boardnet', 'multi_order_board_scan_network', 'cross_stitch_cnn_token_fusion_net', 'neural_decision_forest_boardnet', 'vector_quantized_motif_codebook_net', 'hypercolumn_square_readout_cnn', 'multiplicative_conjunction_convnet', 'empty_square_opportunity_network', 'global_scratchpad_boardnet', 'learnable_pooling_tree_boardnet', 'spatial_film_coordinate_net', 'channel_bilinear_role_mixer', 'evidence_sieve_network', 'ring_shell_recurrent_boardnet', 'rank_file_memory_grid_net', 'negative_class_disentangled_puzzle_head', 'line_piece_crossbar_network', 'near_puzzle_margin_twin_network', 'stripe_selective_mixer_cnn', 'king_zone_evidence_ledger', 'prototype_margin_puzzle_network', 'source_rate_calibrated_objective', 'forcing_certificate_transformer', 'defender_exhaustion_cascade_network', 'causal_piece_derivative_network', 'phase_transition_pressure_network', 'disproof_ledger_puzzle_network', 'motif_tensor_factorization_network', 'tempo_alignment_gate_network', 'puzzle_boundary_twin_encoder', 'critical_square_budget_network', 'legal_reaction_bottleneck_network', 'exchange_soundness_graph_network', 'tactical_program_induction_network', 'counterfactual_defender_dropout_network', 'blocker_pin_lattice_network', 'safe_reply_certificate_verifier', 'latent_reply_entropy_network', 'exchange_then_king_dual_stream', 'tactical_symptom_bayesian_network', 'minimal_edit_puzzle_distance_network', 'source_invariant_puzzle_bottleneck', 'reply_set_contrastive_transformer', 'barrier_cut_puzzle_network', 'tactical_hessian_spectrum_network', 'absorbing_threat_markov_network', 'neural_clause_resolution_puzzle_network', 'piece_liability_gradient_network', 'hierarchical_tactical_option_network', 'cross_defense_consistency_network', 'defender_timing_schedule_network', 'discovered_ray_switchboard_network', 'counterplay_insolvency_ledger', 'pinned_mobility_nullspace_network', 'tactical_effective_resistance_network', 'defender_opportunity_cost_auction_network', 'role_counterfactual_necessity_network', 'phase_specialist_calibration_mixture', 'forced_target_funnel_network', 'tactical_subgoal_automaton_network', 'masked_codec_interaction_curvature_network', 'non_puzzle_score_curl_divergence_bottleneck', 'ray_grammar_edit_distance_network', 'orbit_disagreement_residual_network', 'hall_defect_dual_residual_network', 'credal_temperature_field_network', 'sylvester_tactical_coupling_network', 'schur_complement_defender_network', 'bures_wasserstein_threat_network', 'numerical_range_boundary_network', 'lyapunov_threat_stability_network', 'pfaffian_skew_threat_network', 'padic_ultrametric_threat_network', 'free_probability_r_transform_network', 'williamson_symplectic_threat_network', 'magnus_bch_coupling_series_network', 'riccati_optimal_defense_network', 'clifford_rotor_threat_network', 'tracy_widom_level_spacing_network', 'lindstrom_gessel_viennot_path_network', 'toda_isospectral_flow_network']

RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "hypercolumn_square_readout_cnn"
]

_BESPOKE_MODEL_NAMES = {
    "geometry_conditioned_board_pseudo_likelihood_ratio_network",
    "king_escape_percolation_network",
    "oriented_tactical_sheaf_laplacian",
    "king_shelter_microkernel_network",
    "mobius_piece_constellation_network",
    "one_ply_counterfactual_move_landscape_network",
    "counterfactual_move_delta_spectrum_network",
    "rule_only_counterfactual_move_delta_bottleneck",
    "occupancy_run_length_segment_encoder",
    "patch_mixer_boardnet",
    "piece_plane_gated_cnn",
    "relational_query_algebra_network",
    "sparse_witness_piece_bottleneck_network",
    "specialist_head_cnn",
    "square_color_parity_mixer",
    "tactical_sheaf_curvature_network",
    "tactical_threat_sheaf_network",
    "attack_defense_sheaf_energy_network",
    "attack_hodge_sheaf_tension_network",
    "counterplay_insolvency_ledger",
    "cross_defense_consistency_network",
    "defender_opportunity_cost_auction_network",
    "defender_timing_schedule_network",
    "directed_attack_sheaf_tension_network",
    "file_mirror_tension_sheaf",
    "discovered_ray_switchboard_network",
    "forced_target_funnel_network",
    "hierarchical_tactical_option_network",
    "masked_codec_interaction_curvature_network",
    "masked_board_code_length_surprise_network",
    "non_puzzle_score_curl_divergence_bottleneck",
    "phase_specialist_calibration_mixture",
    "pinned_mobility_nullspace_network",
    "ray_grammar_edit_distance_network",
    "role_counterfactual_necessity_network",
    "tensor_core_square_pair_field_network",
    "tactical_effective_resistance_network",
    "tactical_subgoal_automaton_network",
    "tiny_chess_micronet",
    "variational_board_action_network",
}
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name not in _BESPOKE_MODEL_NAMES
]

RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "geometry_conditioned_board_pseudo_likelihood_ratio_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "king_escape_percolation_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "soft_king_cage_path_bottleneck_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "hall_defect_obligation_matroid_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "threat_topology_betti_bottleneck_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "blocker_pin_lattice_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "safe_reply_certificate_verifier"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "latent_reply_entropy_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "independence_residual_interaction_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "residual_calibration_error_field"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "set_query_attention_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "toda_isospectral_flow_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "multiplicative_conjunction_convnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "empty_square_opportunity_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "global_scratchpad_boardnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "orbit_disagreement_residual_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "hall_defect_dual_residual_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "credal_temperature_field_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "sylvester_tactical_coupling_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "schur_complement_defender_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "bures_wasserstein_threat_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "numerical_range_boundary_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "lyapunov_threat_stability_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "pfaffian_skew_threat_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "padic_ultrametric_threat_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "free_probability_r_transform_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_sheaf_curvature_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "attack_defense_sheaf_energy_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_sheaf_tension_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "entropic_piece_target_transport_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "nuisance_orthogonal_puzzle_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_transport_imbalance_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "king_anchored_material_null_transport_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "piece_target_entropic_transport_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "entropic_chess_geometry_transport_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "ordinal_evidence_ladder_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "ray_language_automaton_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "kinematic_commutator_bottleneck_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "centered_tempo_odd_interventional_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "legal_automorphism_quotient_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "side_canonical_rule_partition_invariant_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "masked_board_code_length_surprise_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "credal_near_puzzle_evidence_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "rule_exact_orbit_bottleneck_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "color_flip_orbit_evidence_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "rule_automorphism_quotient_bottleneck_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tempo_odd_bottleneck_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "king_anchored_euler_interaction_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "non_backtracking_tactical_walk_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "non_puzzle_score_field_bottleneck_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "soft_formal_concept_closure_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "determinantal_tactical_volume_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "harmonic_board_potential_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tropical_constraint_circuit_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "grassmannian_principal_angle_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "matrix_pencil_generalized_spectrum_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "polar_procrustes_alignment_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "multi_scale_dilated_board_mixer_cnn"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "piece_token_cnn_hybrid"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "finite_field_character_sum_board_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "schur_ray_line_algebra_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "bitboard_shift_algebra_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "puzzle_binary_benchmark_challengers"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_bisimulation_puzzle_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "krylov_tactical_subspace_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "adaptive_tactical_resolvent_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_controllability_gramian_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "support_polar_zonotope_certificate_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "support_function_envelope_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "loop_frustration_curvature_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "forcing_response_front_door_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "chess_hypercut_polynomial_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "fisher_geodesic_tension_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "typed_hypergraph_motif_grammar"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "hall_defect_zeta_operator"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "differentiable_chess_fact_lattice"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_radius_filtration"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "chess_mode_tucker_relation_certificate"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_state_bottleneck_inference"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "traced_threat_motif_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "bounded_board_hinge_logic"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "parity_syndrome_puzzle_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "wavelet_scattering_board_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "convex_feasibility_residual_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "rank_quantile_evidence_field_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "oriented_matroid_covector_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "fixed_point_residual_defect_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "baseline_logit_residual_adapter"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "coarse_to_fine_board_residual_pyramid"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "attention_disagreement_residual_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "cross_scale_attention_residual_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "slot_attention_role_binding_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "attention_perturbation_sensitivity_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "kernel_mean_prototype_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tensorsketch_interaction_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "maxout_region_signature_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "spline_board_surface_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "boundary_condition_disagreement_cnn"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "piece_drop_stability_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "row_file_factor_mixer"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "piece_conditioned_hypernetwork_cnn"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "neural_board_cellular_automaton"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "symmetric_difference_twin_encoder"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "prototype_patch_dictionary_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tensor_ring_square_interaction_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "sinkhorn_role_assignment_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "morphological_threat_field_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "invertible_board_coupling_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "sparse_expert_board_router"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "ray_state_space_scan_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "pawn_skeleton_barrier_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "material_phase_low_rank_adapter_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "replicator_payoff_piece_dynamics"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "differentiable_bitboard_boolean_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "orthogonal_board_moment_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "legal_constraint_projection_residual_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "zobrist_kernel_feature_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "low_rank_signed_cut_query_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "early_exit_cascade_boardnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "auxiliary_reconstruction_boardnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "agreement_variance_head_net"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "adapter_sandwich_residual_cnn"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "capsule_motif_boardnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "multi_order_board_scan_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "cross_stitch_cnn_token_fusion_net"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "neural_decision_forest_boardnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "vector_quantized_motif_codebook_net"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "learnable_pooling_tree_boardnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "spatial_film_coordinate_net"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "channel_bilinear_role_mixer"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "evidence_sieve_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "ring_shell_recurrent_boardnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "rank_file_memory_grid_net"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "negative_class_disentangled_puzzle_head"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "line_piece_crossbar_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "near_puzzle_margin_twin_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "stripe_selective_mixer_cnn"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "king_zone_evidence_ledger"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "prototype_margin_puzzle_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "forcing_certificate_transformer"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "causal_piece_derivative_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "phase_transition_pressure_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "disproof_ledger_puzzle_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "motif_tensor_factorization_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tempo_alignment_gate_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "puzzle_boundary_twin_encoder"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "critical_square_budget_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "legal_reaction_bottleneck_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "exchange_soundness_graph_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_program_induction_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "counterfactual_defender_dropout_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "exchange_then_king_dual_stream"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_symptom_bayesian_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "source_invariant_puzzle_bottleneck"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "reply_set_contrastive_transformer"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "minimal_edit_puzzle_distance_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "barrier_cut_puzzle_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tactical_hessian_spectrum_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "absorbing_threat_markov_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "neural_clause_resolution_puzzle_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "piece_liability_gradient_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "williamson_symplectic_threat_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "magnus_bch_coupling_series_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "riccati_optimal_defense_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "clifford_rotor_threat_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "tracy_widom_level_spacing_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "lindstrom_gessel_viennot_path_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "low_displacement_rank_board_operator"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "shallow_wide_residual_boardnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "source_rate_calibrated_objective"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "defender_exhaustion_cascade_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "pivot_trace_elimination_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "local_neighborhood_geometry_network"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "soft_majorization_line_sorter"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "convnext_boardnet"
]
RESEARCH_PACKET_MODEL_NAMES = [
    name for name in RESEARCH_PACKET_MODEL_NAMES if name != "iterative_logit_refinement_cnn"
]
