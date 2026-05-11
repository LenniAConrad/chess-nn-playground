# Paper-Ready Training Status

- Total tasks: `234`
- Completed tasks: `175`
- Remaining tasks: `59`
- ETA: `0s`
- Average observed task time: `7m44s`
- ETA basis: `234` observed task(s), `0` remaining task(s), `1` job(s)
- Dry run: `False`
- Results directory: `results/architecture_scout_2026-05-09`
- Report directory: `reports/architecture_scout_2026-05-09`
- Resume state: `reports/architecture_scout_2026-05-09/state.json`

## Defaults

- Seeds: `42`
- Architecture scales: `base:1`
- Batch-size caps: `base:256,scale_up:192,scale_xl:128`
- Epoch budget: `12`
- Minimum epochs: `6`
- Early-stopping patience: `3`

## Counts

| Status | Count |
|---|---:|
| `artifact_validation_failed` | 1 |
| `completed` | 175 |
| `failed` | 49 |
| `timeout` | 3 |
| `timeout_resume_available` | 6 |

| Kind | Count |
|---|---:|
| `idea` | 234 |

| Architecture Scale | Count |
|---|---:|
| `base` | 234 |

## Open First

- Plan: `reports/architecture_scout_2026-05-09/plan.md`
- Status: `reports/architecture_scout_2026-05-09/status.md`
- State JSON: `reports/architecture_scout_2026-05-09/state.json`
- Event log JSONL: `reports/architecture_scout_2026-05-09/events.jsonl`
- Timeline: `reports/architecture_scout_2026-05-09/timeline.md`
- Logs: `reports/architecture_scout_2026-05-09/logs`
- Generated configs: `reports/architecture_scout_2026-05-09/generated_configs`
- Leaderboard: `results/architecture_scout_2026-05-09/leaderboard.md`
- Seed summary: `results/architecture_scout_2026-05-09/leaderboard_seed_summary.md`
- Training dashboard: `reports/architecture_scout_2026-05-09/training/training_dashboard.md`
- Training dashboard HTML: `reports/architecture_scout_2026-05-09/training/training_dashboard.html`
- Paper PDF report: `reports/architecture_scout_2026-05-09/paper_report.pdf`

## Analysis Jobs

| Job | Return Code | Log |
|---|---:|---|
| `build_paper_report` | `0` | `reports/architecture_scout_2026-05-09/logs/analysis_build_paper_report.log` |
| `compare_results` | `0` | `reports/architecture_scout_2026-05-09/logs/analysis_compare_results.log` |
| `plot_training_results` | `0` | `reports/architecture_scout_2026-05-09/logs/analysis_plot_training_results.log` |

## Speed Snapshot

| Task | Scale | Params | Train Samples/s | Val Samples/s | Total Seconds |
|---|---|---:|---:|---:|---:|
| `idea_i001_chess_operator_basis_classifier_seed42` | `base` | 200393 | 13728.9 | 12326.5 | 193.8 |
| `idea_i002_response_minimax_classifier_seed42` | `base` | 231844 | 2408.2 | 5781.3 | 948.1 |
| `idea_i003_factor_agreement_classifier_seed42` | `base` | 104846 | 14405.3 | 12243.2 | 187.3 |
| `idea_i004_puzzle_obligation_flow_network_seed42` | `base` | 148326 | 2610.8 | 11588.3 | 840.0 |
| `idea_i005_null_move_contrast_puzzle_network_seed42` | `base` | 240578 | 13622.2 | 12117.7 | 196.7 |
| `idea_i006_proof_core_set_verifier_seed42` | `base` | 173587 | 13826.1 | 12107.0 | 193.4 |
| `idea_i007_neural_proof_number_search_seed42` | `base` | 275909 | 5036.7 | 9473.8 | 466.6 |
| `idea_i008_boundary_edit_lagrangian_network_seed42` | `base` | 173764 | 12100.3 | 11956.2 | 216.6 |
| `idea_i009_tactical_equilibrium_network_seed42` | `base` | 176676 | 10725.3 | 11277.2 | 240.1 |
| `idea_i010_rule_consistent_latent_dynamics_seed42` | `base` | 254018 | 13275.7 | 11812.1 | 201.8 |
| `idea_i011_vetoselect_positive_claim_abstention_seed42` | `base` | 501602 | 14283.0 | 11911.4 | 190.0 |
| `idea_i012_dykstra_lcp_seed42` | `base` | 339035 | 4111.2 | 8628.2 | 562.1 |
| `idea_i014_contamination_dro_huber_tail_rejection_seed42` | `base` | 209281 | 14542.7 | 12500.2 | 185.2 |
| `idea_i015_material_locked_tactical_dro_seed42` | `base` | 238529 | 13721.5 | 11811.4 | 196.5 |
| `idea_i017_conditional_surprisal_gate_seed42` | `base` | 218467 | 14954.4 | 13375.8 | 178.3 |
| `idea_i018_oriented_tactical_sheaf_laplacian_seed42` | `base` | 91363 | 2255.2 | 5532.5 | 1010.2 |
| `idea_i019_tactical_sheaf_curvature_network_seed42` | `base` | 111149 | 2751.1 | 6047.8 | 836.5 |
| `idea_i020_attack_defense_sheaf_energy_network_seed42` | `base` | 202790 | 2444.9 | 3971.1 | 972.2 |
| `idea_i025_one_ply_counterfactual_move_landscape_network_seed42` | `base` | 172648 | 2602.8 | 2703.2 | 976.3 |
| `idea_i026_counterfactual_move_delta_spectrum_network_seed42` | `base` | 181675 | 2664.5 | 2710.0 | 957.3 |
| `idea_i027_rule_only_counterfactual_move_delta_bottleneck_seed42` | `base` | 201668 | 2554.6 | 2653.6 | 994.9 |
| `idea_i031_tactical_transport_imbalance_network_seed42` | `base` | 253809 | 9094.9 | 12330.9 | 185.5 |
| `idea_i032_king_anchored_material_null_transport_bottleneck_seed42` | `base` | 24126 | 3092.3 | 4726.2 | 774.3 |
| `idea_i033_piece_target_entropic_transport_bottleneck_seed42` | `base` | 121973 | 2446.0 | 4420.1 | 959.3 |
| `idea_i034_entropic_chess_geometry_transport_network_seed42` | `base` | 77261 | 11795.3 | 12210.1 | 218.8 |
| `idea_i035_ordinal_evidence_ladder_network_seed42` | `base` | 222853 | 15301.8 | 12670.3 | 177.7 |
| `idea_i036_geometry_conditioned_board_pseudo_likelihood_ratio_network_seed42` | `base` | 78273 | 8507.4 | 11885.0 | 287.4 |
| `idea_i037_mobius_piece_constellation_network_seed42` | `base` | 195073 | 14391.6 | 12331.2 | 158.6 |
| `idea_i039_ray_language_automaton_network_seed42` | `base` | 130477 | 1363.0 | 2348.9 | 908.3 |
| `idea_i040_kinematic_commutator_bottleneck_network_seed42` | `base` | 63289 | 1626.5 | 2751.6 | 1450.7 |
| ... | ... | ... | ... | ... | `145` more completed runs |

## Next Tasks

| Task | Kind | Scale | Seed | Status | Source | Run Dir |
|---|---|---|---:|---|---|---|
| `idea_i013_sparse_relation_pursuit_asymmetry_seed42` | `idea` | `base` | `42` | `timeout_resume_available` | `ideas/i013_sparse_relation_pursuit_asymmetry/config.yaml` | `results/architecture_scout_2026-05-09/idea_i013_sparse_relation_pursuit_asymmetry_seed42` |
| `idea_i016_soft_sorting_order_residual_ranker_seed42` | `idea` | `base` | `42` | `timeout` | `ideas/i016_soft_sorting_order_residual_ranker/config.yaml` | `results/architecture_scout_2026-05-09/idea_i016_soft_sorting_order_residual_ranker_seed42` |
| `idea_i021_tactical_sheaf_tension_network_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i021_tactical_sheaf_tension_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i021_tactical_sheaf_tension_network_seed42` |
| `idea_i022_tactical_threat_sheaf_network_seed42` | `idea` | `base` | `42` | `timeout` | `ideas/i022_tactical_threat_sheaf_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i022_tactical_threat_sheaf_network_seed42` |
| `idea_i023_attack_hodge_sheaf_tension_network_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i023_attack_hodge_sheaf_tension_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i023_attack_hodge_sheaf_tension_network_seed42` |
| `idea_i024_directed_attack_sheaf_tension_network_seed42` | `idea` | `base` | `42` | `timeout_resume_available` | `ideas/i024_directed_attack_sheaf_tension_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i024_directed_attack_sheaf_tension_network_seed42` |
| `idea_i028_file_mirror_tension_sheaf_seed42` | `idea` | `base` | `42` | `timeout_resume_available` | `ideas/i028_file_mirror_tension_sheaf/config.yaml` | `results/architecture_scout_2026-05-09/idea_i028_file_mirror_tension_sheaf_seed42` |
| `idea_i029_entropic_piece_target_transport_bottleneck_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i029_entropic_piece_target_transport_bottleneck/config.yaml` | `results/architecture_scout_2026-05-09/idea_i029_entropic_piece_target_transport_bottleneck_seed42` |
| `idea_i030_nuisance_orthogonal_puzzle_bottleneck_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i030_nuisance_orthogonal_puzzle_bottleneck/config.yaml` | `results/architecture_scout_2026-05-09/idea_i030_nuisance_orthogonal_puzzle_bottleneck_seed42` |
| `idea_i038_sparse_witness_piece_bottleneck_network_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i038_sparse_witness_piece_bottleneck_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i038_sparse_witness_piece_bottleneck_network_seed42` |
| `idea_i053_hall_defect_obligation_matroid_network_seed42` | `idea` | `base` | `42` | `timeout_resume_available` | `ideas/i053_hall_defect_obligation_matroid_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i053_hall_defect_obligation_matroid_network_seed42` |
| `idea_i055_non_backtracking_tactical_walk_network_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i055_non_backtracking_tactical_walk_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i055_non_backtracking_tactical_walk_network_seed42` |
| `idea_i058_determinantal_tactical_volume_bottleneck_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i058_determinantal_tactical_volume_bottleneck/config.yaml` | `results/architecture_scout_2026-05-09/idea_i058_determinantal_tactical_volume_bottleneck_seed42` |
| `idea_i061_grassmannian_principal_angle_bottleneck_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i061_grassmannian_principal_angle_bottleneck/config.yaml` | `results/architecture_scout_2026-05-09/idea_i061_grassmannian_principal_angle_bottleneck_seed42` |
| `idea_i063_polar_procrustes_alignment_bottleneck_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i063_polar_procrustes_alignment_bottleneck/config.yaml` | `results/architecture_scout_2026-05-09/idea_i063_polar_procrustes_alignment_bottleneck_seed42` |
| `idea_i067_finite_field_character_sum_board_network_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i067_finite_field_character_sum_board_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i067_finite_field_character_sum_board_network_seed42` |
| `idea_i068_schur_ray_line_algebra_network_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i068_schur_ray_line_algebra_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i068_schur_ray_line_algebra_network_seed42` |
| `idea_i073_tiny_chess_micronet_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i073_tiny_chess_micronet/config.yaml` | `results/architecture_scout_2026-05-09/idea_i073_tiny_chess_micronet_seed42` |
| `idea_i076_krylov_tactical_subspace_network_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i076_krylov_tactical_subspace_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i076_krylov_tactical_subspace_network_seed42` |
| `idea_i077_adaptive_tactical_resolvent_network_seed42` | `idea` | `base` | `42` | `failed` | `ideas/i077_adaptive_tactical_resolvent_network/config.yaml` | `results/architecture_scout_2026-05-09/idea_i077_adaptive_tactical_resolvent_network_seed42` |

## Needs Attention

| Task | Status | Log | Messages |
|---|---|---|---|
| `idea_i013_sparse_relation_pursuit_asymmetry_seed42` | `timeout_resume_available` | `reports/architecture_scout_2026-05-09/logs/idea_i013_sparse_relation_pursuit_asymmetry_seed42_attempt1.log` | Timed out after 60.0 minutes |
| `idea_i016_soft_sorting_order_residual_ranker_seed42` | `timeout` | `reports/architecture_scout_2026-05-09/logs/idea_i016_soft_sorting_order_residual_ranker_seed42_attempt1.log` | Timed out after 60.0 minutes |
| `idea_i021_tactical_sheaf_tension_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i021_tactical_sheaf_tension_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i022_tactical_threat_sheaf_network_seed42` | `timeout` | `reports/architecture_scout_2026-05-09/logs/idea_i022_tactical_threat_sheaf_network_seed42_attempt1.log` | Timed out after 60.0 minutes |
| `idea_i023_attack_hodge_sheaf_tension_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i023_attack_hodge_sheaf_tension_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i024_directed_attack_sheaf_tension_network_seed42` | `timeout_resume_available` | `reports/architecture_scout_2026-05-09/logs/idea_i024_directed_attack_sheaf_tension_network_seed42_attempt1.log` | Timed out after 60.0 minutes |
| `idea_i028_file_mirror_tension_sheaf_seed42` | `timeout_resume_available` | `reports/architecture_scout_2026-05-09/logs/idea_i028_file_mirror_tension_sheaf_seed42_attempt1.log` | Timed out after 60.0 minutes |
| `idea_i029_entropic_piece_target_transport_bottleneck_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i029_entropic_piece_target_transport_bottleneck_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i030_nuisance_orthogonal_puzzle_bottleneck_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i030_nuisance_orthogonal_puzzle_bottleneck_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i038_sparse_witness_piece_bottleneck_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i038_sparse_witness_piece_bottleneck_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i053_hall_defect_obligation_matroid_network_seed42` | `timeout_resume_available` | `reports/architecture_scout_2026-05-09/logs/idea_i053_hall_defect_obligation_matroid_network_seed42_attempt1.log` | Timed out after 60.0 minutes |
| `idea_i055_non_backtracking_tactical_walk_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i055_non_backtracking_tactical_walk_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i058_determinantal_tactical_volume_bottleneck_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i058_determinantal_tactical_volume_bottleneck_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i061_grassmannian_principal_angle_bottleneck_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i061_grassmannian_principal_angle_bottleneck_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i063_polar_procrustes_alignment_bottleneck_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i063_polar_procrustes_alignment_bottleneck_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i067_finite_field_character_sum_board_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i067_finite_field_character_sum_board_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i068_schur_ray_line_algebra_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i068_schur_ray_line_algebra_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i073_tiny_chess_micronet_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i073_tiny_chess_micronet_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i076_krylov_tactical_subspace_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i076_krylov_tactical_subspace_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i077_adaptive_tactical_resolvent_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i077_adaptive_tactical_resolvent_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i078_tactical_controllability_gramian_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i078_tactical_controllability_gramian_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i079_support_polar_zonotope_certificate_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i079_support_polar_zonotope_certificate_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i080_loop_frustration_curvature_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i080_loop_frustration_curvature_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i081_forcing_response_front_door_bottleneck_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i081_forcing_response_front_door_bottleneck_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i082_chess_hypercut_polynomial_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i082_chess_hypercut_polynomial_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i084_typed_hypergraph_motif_grammar_seed42` | `timeout_resume_available` | `reports/architecture_scout_2026-05-09/logs/idea_i084_typed_hypergraph_motif_grammar_seed42_attempt1.log` | Timed out after 60.0 minutes |
| `idea_i085_hall_defect_zeta_operator_seed42` | `timeout` | `reports/architecture_scout_2026-05-09/logs/idea_i085_hall_defect_zeta_operator_seed42_attempt1.log` | Timed out after 60.0 minutes |
| `idea_i087_tactical_radius_filtration_seed42` | `timeout_resume_available` | `reports/architecture_scout_2026-05-09/logs/idea_i087_tactical_radius_filtration_seed42_attempt1.log` | Timed out after 60.0 minutes |
| `idea_i089_bounded_board_hinge_logic_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i089_bounded_board_hinge_logic_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i090_chess_mode_tucker_relation_certificate_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i090_chess_mode_tucker_relation_certificate_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i091_tactical_state_bottleneck_inference_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i091_tactical_state_bottleneck_inference_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i117_prototype_patch_dictionary_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i117_prototype_patch_dictionary_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i123_sparse_expert_board_router_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i123_sparse_expert_board_router_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i124_local_neighborhood_geometry_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i124_local_neighborhood_geometry_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i125_ray_state_space_scan_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i125_ray_state_space_scan_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i151_auxiliary_reconstruction_boardnet_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i151_auxiliary_reconstruction_boardnet_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i156_multi_order_board_scan_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i156_multi_order_board_scan_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i157_cross_stitch_cnn_token_fusion_net_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i157_cross_stitch_cnn_token_fusion_net_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i161_multiplicative_conjunction_convnet_seed42` | `artifact_validation_failed` | `reports/architecture_scout_2026-05-09/logs/idea_i161_multiplicative_conjunction_convnet_seed42_attempt1.log` | ERROR: missing required artifact: calibration_plot.png |
| `idea_i165_spatial_film_coordinate_net_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i165_spatial_film_coordinate_net_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i166_channel_bilinear_role_mixer_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i166_channel_bilinear_role_mixer_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i168_ring_shell_recurrent_boardnet_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i168_ring_shell_recurrent_boardnet_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i180_phase_transition_pressure_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i180_phase_transition_pressure_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i186_legal_reaction_bottleneck_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i186_legal_reaction_bottleneck_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i187_exchange_soundness_graph_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i187_exchange_soundness_graph_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i190_blocker_pin_lattice_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i190_blocker_pin_lattice_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i221_sylvester_tactical_coupling_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i221_sylvester_tactical_coupling_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i222_schur_complement_defender_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i222_schur_complement_defender_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i223_bures_wasserstein_threat_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i223_bures_wasserstein_threat_network_seed42_attempt1.log` | Training command failed with return code 1 |
| `idea_i224_numerical_range_boundary_network_seed42` | `failed` | `reports/architecture_scout_2026-05-09/logs/idea_i224_numerical_range_boundary_network_seed42_attempt1.log` | Training command failed with return code 1 |
| ... | ... | ... | `9` more tasks need attention or are pending. |

## Resume Command

Rerun the same command after an interruption. Completed tasks stay completed, and unfinished tasks use the same fixed run directories.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py --results-dir results/architecture_scout_2026-05-09 --report-dir reports/architecture_scout_2026-05-09 --state-path reports/architecture_scout_2026-05-09/state.json --logs-dir reports/architecture_scout_2026-05-09/logs --generated-config-dir reports/architecture_scout_2026-05-09/generated_configs --seeds 42 --scale-variants base:1 --batch-size-caps base:256,scale_up:192,scale_xl:128 --epochs 12 --min-epochs 6 --patience 3 --jobs 1 --gpu-ids 0 --timeout-minutes 60.0 --no-benchmarks
```
