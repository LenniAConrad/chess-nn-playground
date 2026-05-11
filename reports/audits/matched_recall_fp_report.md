# Matched-recall false-positive and worst-slice report

- results-root: `_scout_combined_view`
- runs analyzed: 182 (skipped 52 non-puzzle_binary or missing predictions)
- recall targets: [0.8, 0.85]
- ranking: lower `near_puzzle_fp_rate` at recall 0.8 is better

## Top 25 by lowest near-puzzle FP rate at recall 0.8

| run | n_pos | n_near | recall | precision | total_FP | near_FP | near_FP_rate | far_FP_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `idea_i193_exchange_then_king_dual_stream_seed42` | 7055 | 7240 | 0.800 | 0.807 | 1350 | 927 | 0.128 | 0.059 |
| `idea_i024_directed_attack_sheaf_tension_network_seed42` | 7104 | 7012 | 0.800 | 0.801 | 1409 | 982 | 0.140 | 0.059 |
| `idea_i018_oriented_tactical_sheaf_laplacian_seed42` | 7055 | 7240 | 0.800 | 0.782 | 1575 | 1083 | 0.150 | 0.068 |
| `idea_i011_vetoselect_positive_claim_abstention_seed42` | 7055 | 7240 | 0.800 | 0.781 | 1581 | 1111 | 0.153 | 0.065 |
| `idea_i192_latent_reply_entropy_network_seed42` | 7055 | 7240 | 0.800 | 0.781 | 1580 | 1132 | 0.156 | 0.062 |
| `idea_i087_tactical_radius_filtration_seed42` | 7104 | 7012 | 0.800 | 0.781 | 1596 | 1099 | 0.157 | 0.069 |
| `idea_i188_tactical_program_induction_network_seed42` | 7055 | 7240 | 0.800 | 0.777 | 1620 | 1150 | 0.159 | 0.065 |
| `idea_i048_rule_automorphism_quotient_bottleneck_network_seed42` | 7055 | 7240 | 0.800 | 0.777 | 1619 | 1154 | 0.159 | 0.065 |
| `idea_i191_safe_reply_certificate_verifier_seed42` | 7055 | 7240 | 0.800 | 0.778 | 1615 | 1155 | 0.160 | 0.064 |
| `idea_i042_legal_automorphism_quotient_network_seed42` | 7055 | 7240 | 0.800 | 0.768 | 1708 | 1208 | 0.167 | 0.069 |
| `idea_i013_sparse_relation_pursuit_asymmetry_seed42` | 7104 | 7012 | 0.800 | 0.768 | 1715 | 1178 | 0.168 | 0.075 |
| `idea_i147_specialist_head_cnn_seed42` | 7055 | 7240 | 0.800 | 0.764 | 1747 | 1247 | 0.172 | 0.069 |
| `idea_i046_rule_exact_orbit_bottleneck_network_seed42` | 7055 | 7240 | 0.800 | 0.761 | 1769 | 1273 | 0.176 | 0.069 |
| `idea_i033_piece_target_entropic_transport_bottleneck_seed42` | 7055 | 7240 | 0.800 | 0.759 | 1791 | 1279 | 0.177 | 0.071 |
| `idea_i012_dykstra_lcp_seed42` | 7055 | 7240 | 0.800 | 0.755 | 1830 | 1294 | 0.179 | 0.074 |
| `idea_i100_independence_residual_interaction_network_seed42` | 7055 | 7240 | 0.800 | 0.760 | 1786 | 1298 | 0.179 | 0.068 |
| `idea_i145_piece_plane_gated_cnn_seed42` | 7055 | 7240 | 0.800 | 0.755 | 1827 | 1305 | 0.180 | 0.072 |
| `idea_i218_orbit_disagreement_residual_network_seed42` | 7055 | 7240 | 0.800 | 0.748 | 1898 | 1360 | 0.188 | 0.075 |
| `idea_i095_rank_quantile_evidence_field_network_seed42` | 7055 | 7240 | 0.800 | 0.742 | 1962 | 1417 | 0.196 | 0.076 |
| `idea_i086_differentiable_chess_fact_lattice_seed42` | 7055 | 7240 | 0.800 | 0.727 | 2121 | 1429 | 0.197 | 0.096 |
| `idea_i043_side_canonical_rule_partition_invariant_bottleneck_seed42` | 7055 | 7240 | 0.800 | 0.735 | 2030 | 1430 | 0.198 | 0.083 |
| `idea_i189_counterfactual_defender_dropout_network_seed42` | 7055 | 7240 | 0.800 | 0.737 | 2015 | 1452 | 0.201 | 0.078 |
| `idea_i099_coarse_to_fine_board_residual_pyramid_seed42` | 7055 | 7240 | 0.800 | 0.739 | 1991 | 1466 | 0.202 | 0.073 |
| `idea_i130_material_phase_low_rank_adapter_network_seed42` | 7055 | 7240 | 0.800 | 0.734 | 2049 | 1478 | 0.204 | 0.079 |
| `idea_i131_replicator_payoff_piece_dynamics_seed42` | 7055 | 7240 | 0.800 | 0.732 | 2068 | 1479 | 0.204 | 0.082 |

## Same table at recall 0.85

| run | recall | precision | total_FP | near_FP | near_FP_rate | far_FP_rate |
|---|---:|---:|---:|---:|---:|---:|
| `idea_i193_exchange_then_king_dual_stream_seed42` | 0.850 | 0.776 | 1730 | 1198 | 0.165 | 0.074 |
| `idea_i024_directed_attack_sheaf_tension_network_seed42` | 0.850 | 0.771 | 1792 | 1265 | 0.180 | 0.073 |
| `idea_i018_oriented_tactical_sheaf_laplacian_seed42` | 0.850 | 0.756 | 1940 | 1344 | 0.186 | 0.083 |
| `idea_i087_tactical_radius_filtration_seed42` | 0.850 | 0.752 | 1991 | 1380 | 0.197 | 0.085 |
| `idea_i192_latent_reply_entropy_network_seed42` | 0.850 | 0.749 | 2009 | 1432 | 0.198 | 0.080 |
| `idea_i011_vetoselect_positive_claim_abstention_seed42` | 0.850 | 0.744 | 2062 | 1445 | 0.200 | 0.086 |
| `idea_i191_safe_reply_certificate_verifier_seed42` | 0.850 | 0.748 | 2016 | 1447 | 0.200 | 0.079 |
| `idea_i048_rule_automorphism_quotient_bottleneck_network_seed42` | 0.850 | 0.749 | 2006 | 1449 | 0.200 | 0.077 |
| `idea_i042_legal_automorphism_quotient_network_seed42` | 0.850 | 0.739 | 2123 | 1505 | 0.208 | 0.086 |
| `idea_i013_sparse_relation_pursuit_asymmetry_seed42` | 0.850 | 0.736 | 2171 | 1490 | 0.212 | 0.095 |
| `idea_i188_tactical_program_induction_network_seed42` | 0.850 | 0.732 | 2192 | 1543 | 0.213 | 0.090 |
| `idea_i046_rule_exact_orbit_bottleneck_network_seed42` | 0.850 | 0.732 | 2191 | 1553 | 0.215 | 0.089 |
| `idea_i033_piece_target_entropic_transport_bottleneck_seed42` | 0.850 | 0.727 | 2248 | 1600 | 0.221 | 0.090 |
| `idea_i100_independence_residual_interaction_network_seed42` | 0.850 | 0.729 | 2230 | 1630 | 0.225 | 0.083 |
| `idea_i147_specialist_head_cnn_seed42` | 0.850 | 0.727 | 2257 | 1637 | 0.226 | 0.086 |
| `idea_i012_dykstra_lcp_seed42` | 0.850 | 0.721 | 2326 | 1665 | 0.230 | 0.092 |
| `idea_i145_piece_plane_gated_cnn_seed42` | 0.850 | 0.721 | 2317 | 1665 | 0.230 | 0.090 |
| `idea_i218_orbit_disagreement_residual_network_seed42` | 0.850 | 0.716 | 2376 | 1695 | 0.234 | 0.095 |
| `idea_i084_typed_hypergraph_motif_grammar_seed42` | 0.850 | 0.714 | 2419 | 1700 | 0.242 | 0.100 |
| `idea_i086_differentiable_chess_fact_lattice_seed42` | 0.850 | 0.691 | 2688 | 1771 | 0.245 | 0.127 |
| `idea_i095_rank_quantile_evidence_field_network_seed42` | 0.850 | 0.708 | 2476 | 1780 | 0.246 | 0.097 |
| `idea_i099_coarse_to_fine_board_residual_pyramid_seed42` | 0.850 | 0.707 | 2483 | 1821 | 0.252 | 0.092 |
| `idea_i043_side_canonical_rule_partition_invariant_bottleneck_seed42` | 0.850 | 0.697 | 2601 | 1824 | 0.252 | 0.108 |
| `idea_i189_counterfactual_defender_dropout_network_seed42` | 0.850 | 0.700 | 2573 | 1843 | 0.255 | 0.101 |
| `idea_i044_masked_board_code_length_surprise_network_seed42` | 0.850 | 0.697 | 2610 | 1849 | 0.255 | 0.106 |

## Worst slices (top 5 per run, sample size >= 50)

### `idea_i193_exchange_then_king_dual_stream_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_eval_bucket | equal | 3485 | 0.770 |
| crtk_difficulty | hard | 4181 | 0.782 |
| crtk_eval_bucket | slight_white | 3406 | 0.805 |
| crtk_difficulty | very_hard | 5860 | 0.808 |
| crtk_phase | endgame | 4338 | 0.820 |

### `idea_i024_directed_attack_sheaf_tension_network_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_eval_bucket | equal | 3482 | 0.747 |
| crtk_difficulty | hard | 4256 | 0.763 |
| crtk_tactic_motifs | promotion | 578 | 0.799 |
| crtk_tactic_motifs | underpromotion | 578 | 0.799 |
| crtk_difficulty | very_hard | 5667 | 0.803 |

### `idea_i018_oriented_tactical_sheaf_laplacian_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_eval_bucket | equal | 3485 | 0.743 |
| crtk_difficulty | hard | 4181 | 0.769 |
| crtk_tactic_motifs | mate_in_1 | 1050 | 0.775 |
| crtk_difficulty | very_hard | 5860 | 0.795 |
| crtk_tactic_motifs | promotion | 632 | 0.801 |

### `idea_i011_vetoselect_positive_claim_abstention_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_eval_bucket | equal | 3485 | 0.761 |
| crtk_difficulty | hard | 4181 | 0.781 |
| crtk_difficulty | very_hard | 5860 | 0.794 |
| crtk_tactic_motifs | mate_in_1 | 1050 | 0.801 |
| crtk_tactic_motifs | promotion | 632 | 0.804 |

### `idea_i192_latent_reply_entropy_network_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_tactic_motifs | mate_in_1 | 1050 | 0.753 |
| crtk_eval_bucket | equal | 3485 | 0.756 |
| crtk_difficulty | hard | 4181 | 0.771 |
| crtk_difficulty | very_hard | 5860 | 0.793 |
| crtk_tactic_motifs | promotion | 632 | 0.801 |

### `idea_i087_tactical_radius_filtration_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_eval_bucket | equal | 3482 | 0.736 |
| crtk_difficulty | hard | 4256 | 0.748 |
| crtk_tactic_motifs | promotion | 578 | 0.780 |
| crtk_tactic_motifs | underpromotion | 578 | 0.780 |
| crtk_tactic_motifs | mate_in_1 | 966 | 0.795 |

### `idea_i188_tactical_program_induction_network_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_eval_bucket | equal | 3485 | 0.752 |
| crtk_difficulty | hard | 4181 | 0.766 |
| crtk_tactic_motifs | promotion | 632 | 0.801 |
| crtk_tactic_motifs | underpromotion | 632 | 0.801 |
| crtk_difficulty | very_hard | 5860 | 0.804 |

### `idea_i048_rule_automorphism_quotient_bottleneck_network_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_eval_bucket | equal | 3485 | 0.750 |
| crtk_difficulty | hard | 4181 | 0.773 |
| crtk_tactic_motifs | mate_in_1 | 1050 | 0.785 |
| crtk_difficulty | very_hard | 5860 | 0.792 |
| crtk_tactic_motifs | promotion | 632 | 0.802 |

### `idea_i191_safe_reply_certificate_verifier_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_eval_bucket | equal | 3485 | 0.734 |
| crtk_difficulty | hard | 4181 | 0.758 |
| crtk_tactic_motifs | mate_in_1 | 1050 | 0.778 |
| crtk_difficulty | very_hard | 5860 | 0.790 |
| crtk_eval_bucket | slight_white | 3406 | 0.803 |

### `idea_i042_legal_automorphism_quotient_network_seed42`  (slice acc evaluated at recall 0.8)

| slice | value | n | accuracy |
|---|---|---:|---:|
| crtk_eval_bucket | equal | 3485 | 0.743 |
| crtk_difficulty | hard | 4181 | 0.758 |
| crtk_tactic_motifs | promotion | 632 | 0.783 |
| crtk_tactic_motifs | underpromotion | 632 | 0.783 |
| crtk_tactic_motifs | mate_in_1 | 1050 | 0.788 |


## Promotion / underpromotion tactic-motif slices (top 25 by near-puzzle FP rejection)

| run | motif | n | n_near_neg | accuracy@recall | near_FP | near_FP_rate |
|---|---|---:|---:|---:|---:|---:|
| `idea_i024_directed_attack_sheaf_tension_network_seed42` | promotion | 578 | 255 | 0.799 | 24 | 0.094 |
| `idea_i024_directed_attack_sheaf_tension_network_seed42` | underpromotion | 578 | 255 | 0.799 | 24 | 0.094 |
| `idea_i191_safe_reply_certificate_verifier_seed42` | promotion | 632 | 271 | 0.807 | 27 | 0.100 |
| `idea_i191_safe_reply_certificate_verifier_seed42` | underpromotion | 632 | 271 | 0.807 | 27 | 0.100 |
| `idea_i013_sparse_relation_pursuit_asymmetry_seed42` | promotion | 578 | 255 | 0.818 | 26 | 0.102 |
| `idea_i013_sparse_relation_pursuit_asymmetry_seed42` | underpromotion | 578 | 255 | 0.818 | 26 | 0.102 |
| `idea_i193_exchange_then_king_dual_stream_seed42` | promotion | 632 | 271 | 0.837 | 28 | 0.103 |
| `idea_i193_exchange_then_king_dual_stream_seed42` | underpromotion | 632 | 271 | 0.837 | 28 | 0.103 |
| `idea_i192_latent_reply_entropy_network_seed42` | promotion | 632 | 271 | 0.801 | 34 | 0.125 |
| `idea_i192_latent_reply_entropy_network_seed42` | underpromotion | 632 | 271 | 0.801 | 34 | 0.125 |
| `idea_i018_oriented_tactical_sheaf_laplacian_seed42` | promotion | 632 | 271 | 0.801 | 35 | 0.129 |
| `idea_i018_oriented_tactical_sheaf_laplacian_seed42` | underpromotion | 632 | 271 | 0.801 | 35 | 0.129 |
| `idea_i033_piece_target_entropic_transport_bottleneck_seed42` | promotion | 632 | 271 | 0.809 | 37 | 0.137 |
| `idea_i033_piece_target_entropic_transport_bottleneck_seed42` | underpromotion | 632 | 271 | 0.809 | 37 | 0.137 |
| `idea_i147_specialist_head_cnn_seed42` | promotion | 632 | 271 | 0.797 | 39 | 0.144 |
| `idea_i147_specialist_head_cnn_seed42` | underpromotion | 632 | 271 | 0.797 | 39 | 0.144 |
| `idea_i042_legal_automorphism_quotient_network_seed42` | promotion | 632 | 271 | 0.783 | 40 | 0.148 |
| `idea_i042_legal_automorphism_quotient_network_seed42` | underpromotion | 632 | 271 | 0.783 | 40 | 0.148 |
| `idea_i173_stripe_selective_mixer_cnn_seed42` | promotion | 632 | 271 | 0.796 | 40 | 0.148 |
| `idea_i173_stripe_selective_mixer_cnn_seed42` | underpromotion | 632 | 271 | 0.796 | 40 | 0.148 |
| `idea_i130_material_phase_low_rank_adapter_network_seed42` | promotion | 632 | 271 | 0.785 | 41 | 0.151 |
| `idea_i130_material_phase_low_rank_adapter_network_seed42` | underpromotion | 632 | 271 | 0.785 | 41 | 0.151 |
| `idea_i188_tactical_program_induction_network_seed42` | promotion | 632 | 271 | 0.801 | 44 | 0.162 |
| `idea_i188_tactical_program_induction_network_seed42` | underpromotion | 632 | 271 | 0.801 | 44 | 0.162 |
| `idea_i048_rule_automorphism_quotient_bottleneck_network_seed42` | promotion | 632 | 271 | 0.802 | 44 | 0.162 |
