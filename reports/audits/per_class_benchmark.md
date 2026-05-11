# Per-class benchmark report (puzzle_binary)

- results-root: `_scout_combined_view`
- recall target for accuracy@recall metric: **0.8**
- runs analyzed: 182 (skipped 52 non-puzzle_binary)
- groups (3+ seeds): 182

All numeric cells are 3-seed mean (± std). Higher PR AUC is better.
Per-slice PR AUC is computed restricted to that slice's rows.


## Matrix 1 — model × difficulty (test PR AUC, restricted to slice)

| Group | overall | very_easy | easy | medium | hard | very_hard |
|---|---:|---:|---:|---:|---:|---:|
| `idea_i193_exchange_then_king_dual_stream` | 0.876 ± 0.000 | 0.573 ± 0.000 | 0.661 ± 0.000 | 0.711 ± 0.000 | 0.792 ± 0.000 | 0.924 ± 0.000 |
| `idea_i024_directed_attack_sheaf_tension_network` | 0.871 ± 0.000 | 0.656 ± 0.000 | 0.698 ± 0.000 | 0.703 ± 0.000 | 0.762 ± 0.000 | 0.927 ± 0.000 |
| `idea_i087_tactical_radius_filtration` | 0.864 ± 0.000 | 0.548 ± 0.000 | 0.608 ± 0.000 | 0.636 ± 0.000 | 0.753 ± 0.000 | 0.926 ± 0.000 |
| `idea_i048_rule_automorphism_quotient_bottleneck_network` | 0.861 ± 0.000 | 0.518 ± 0.000 | 0.569 ± 0.000 | 0.635 ± 0.000 | 0.770 ± 0.000 | 0.920 ± 0.000 |
| `idea_i018_oriented_tactical_sheaf_laplacian` | 0.861 ± 0.000 | 0.453 ± 0.000 | 0.596 ± 0.000 | 0.668 ± 0.000 | 0.773 ± 0.000 | 0.915 ± 0.000 |
| `idea_i188_tactical_program_induction_network` | 0.861 ± 0.000 | 0.460 ± 0.000 | 0.497 ± 0.000 | 0.600 ± 0.000 | 0.778 ± 0.000 | 0.921 ± 0.000 |
| `idea_i011_vetoselect_positive_claim_abstention` | 0.858 ± 0.000 | 0.432 ± 0.000 | 0.524 ± 0.000 | 0.601 ± 0.000 | 0.779 ± 0.000 | 0.917 ± 0.000 |
| `idea_i013_sparse_relation_pursuit_asymmetry` | 0.856 ± 0.000 | 0.571 ± 0.000 | 0.487 ± 0.000 | 0.596 ± 0.000 | 0.752 ± 0.000 | 0.926 ± 0.000 |
| `idea_i192_latent_reply_entropy_network` | 0.855 ± 0.000 | 0.473 ± 0.000 | 0.569 ± 0.000 | 0.631 ± 0.000 | 0.763 ± 0.000 | 0.910 ± 0.000 |
| `idea_i191_safe_reply_certificate_verifier` | 0.852 ± 0.000 | 0.499 ± 0.000 | 0.612 ± 0.000 | 0.636 ± 0.000 | 0.736 ± 0.000 | 0.910 ± 0.000 |
| `idea_i042_legal_automorphism_quotient_network` | 0.852 ± 0.000 | 0.406 ± 0.000 | 0.511 ± 0.000 | 0.575 ± 0.000 | 0.749 ± 0.000 | 0.915 ± 0.000 |
| `idea_i147_specialist_head_cnn` | 0.851 ± 0.000 | 0.497 ± 0.000 | 0.469 ± 0.000 | 0.607 ± 0.000 | 0.753 ± 0.000 | 0.913 ± 0.000 |

## Matrix 2 — model × phase (test PR AUC)

| Group | overall | opening | middlegame | endgame |
|---|---:|---:|---:|---:|
| `idea_i193_exchange_then_king_dual_stream` | 0.876 ± 0.000 | 0.862 ± 0.000 | 0.891 ± 0.000 | 0.851 ± 0.000 |
| `idea_i024_directed_attack_sheaf_tension_network` | 0.871 ± 0.000 | 0.864 ± 0.000 | 0.886 ± 0.000 | 0.840 ± 0.000 |
| `idea_i087_tactical_radius_filtration` | 0.864 ± 0.000 | 0.862 ± 0.000 | 0.881 ± 0.000 | 0.829 ± 0.000 |
| `idea_i048_rule_automorphism_quotient_bottleneck_network` | 0.861 ± 0.000 | 0.828 ± 0.000 | 0.879 ± 0.000 | 0.850 ± 0.000 |
| `idea_i018_oriented_tactical_sheaf_laplacian` | 0.861 ± 0.000 | 0.850 ± 0.000 | 0.872 ± 0.000 | 0.845 ± 0.000 |
| `idea_i188_tactical_program_induction_network` | 0.861 ± 0.000 | 0.840 ± 0.000 | 0.872 ± 0.000 | 0.854 ± 0.000 |
| `idea_i011_vetoselect_positive_claim_abstention` | 0.858 ± 0.000 | 0.846 ± 0.000 | 0.871 ± 0.000 | 0.837 ± 0.000 |
| `idea_i013_sparse_relation_pursuit_asymmetry` | 0.856 ± 0.000 | 0.843 ± 0.000 | 0.866 ± 0.000 | 0.846 ± 0.000 |
| `idea_i192_latent_reply_entropy_network` | 0.855 ± 0.000 | 0.832 ± 0.000 | 0.870 ± 0.000 | 0.845 ± 0.000 |
| `idea_i191_safe_reply_certificate_verifier` | 0.852 ± 0.000 | 0.814 ± 0.000 | 0.871 ± 0.000 | 0.843 ± 0.000 |
| `idea_i042_legal_automorphism_quotient_network` | 0.852 ± 0.000 | 0.830 ± 0.000 | 0.868 ± 0.000 | 0.831 ± 0.000 |
| `idea_i147_specialist_head_cnn` | 0.851 ± 0.000 | 0.830 ± 0.000 | 0.864 ± 0.000 | 0.840 ± 0.000 |

## Matrix 3 — model × eval_bucket (test PR AUC)

| Group | overall | crushing_white | winning_white | clear_white | slight_white | equal | slight_black | clear_black | winning_black | crushing_black |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idea_i193_exchange_then_king_dual_stream` | 0.876 ± 0.000 | 0.970 ± 0.000 | 0.964 ± 0.000 | 0.914 ± 0.000 | 0.833 ± 0.000 | 0.817 ± 0.000 | 0.855 ± 0.000 | 0.917 ± 0.000 | 0.960 ± 0.000 | 0.961 ± 0.000 |
| `idea_i024_directed_attack_sheaf_tension_network` | 0.871 ± 0.000 | 0.937 ± 0.000 | 0.970 ± 0.000 | 0.921 ± 0.000 | 0.837 ± 0.000 | 0.793 ± 0.000 | 0.859 ± 0.000 | 0.900 ± 0.000 | 0.985 ± 0.000 | 0.896 ± 0.000 |
| `idea_i087_tactical_radius_filtration` | 0.864 ± 0.000 | 0.930 ± 0.000 | 0.957 ± 0.000 | 0.922 ± 0.000 | 0.823 ± 0.000 | 0.791 ± 0.000 | 0.849 ± 0.000 | 0.891 ± 0.000 | 0.976 ± 0.000 | 0.919 ± 0.000 |
| `idea_i048_rule_automorphism_quotient_bottleneck_network` | 0.861 ± 0.000 | 0.989 ± 0.000 | 0.975 ± 0.000 | 0.904 ± 0.000 | 0.817 ± 0.000 | 0.795 ± 0.000 | 0.827 ± 0.000 | 0.906 ± 0.000 | 0.970 ± 0.000 | 0.972 ± 0.000 |
| `idea_i018_oriented_tactical_sheaf_laplacian` | 0.861 ± 0.000 | 0.977 ± 0.000 | 0.954 ± 0.000 | 0.900 ± 0.000 | 0.825 ± 0.000 | 0.799 ± 0.000 | 0.831 ± 0.000 | 0.903 ± 0.000 | 0.968 ± 0.000 | 0.970 ± 0.000 |
| `idea_i188_tactical_program_induction_network` | 0.861 ± 0.000 | 0.971 ± 0.000 | 0.957 ± 0.000 | 0.895 ± 0.000 | 0.822 ± 0.000 | 0.801 ± 0.000 | 0.833 ± 0.000 | 0.902 ± 0.000 | 0.966 ± 0.000 | 0.971 ± 0.000 |
| `idea_i011_vetoselect_positive_claim_abstention` | 0.858 ± 0.000 | 0.981 ± 0.000 | 0.974 ± 0.000 | 0.896 ± 0.000 | 0.818 ± 0.000 | 0.802 ± 0.000 | 0.832 ± 0.000 | 0.896 ± 0.000 | 0.965 ± 0.000 | 0.972 ± 0.000 |
| `idea_i013_sparse_relation_pursuit_asymmetry` | 0.856 ± 0.000 | 0.945 ± 0.000 | 0.970 ± 0.000 | 0.907 ± 0.000 | 0.810 ± 0.000 | 0.792 ± 0.000 | 0.827 ± 0.000 | 0.891 ± 0.000 | 0.978 ± 0.000 | 0.925 ± 0.000 |
| `idea_i192_latent_reply_entropy_network` | 0.855 ± 0.000 | 0.984 ± 0.000 | 0.971 ± 0.000 | 0.900 ± 0.000 | 0.807 ± 0.000 | 0.792 ± 0.000 | 0.820 ± 0.000 | 0.899 ± 0.000 | 0.950 ± 0.000 | 0.974 ± 0.000 |
| `idea_i191_safe_reply_certificate_verifier` | 0.852 ± 0.000 | 0.982 ± 0.000 | 0.951 ± 0.000 | 0.894 ± 0.000 | 0.805 ± 0.000 | 0.767 ± 0.000 | 0.825 ± 0.000 | 0.905 ± 0.000 | 0.962 ± 0.000 | 0.968 ± 0.000 |
| `idea_i042_legal_automorphism_quotient_network` | 0.852 ± 0.000 | 0.978 ± 0.000 | 0.967 ± 0.000 | 0.890 ± 0.000 | 0.798 ± 0.000 | 0.776 ± 0.000 | 0.822 ± 0.000 | 0.904 ± 0.000 | 0.958 ± 0.000 | 0.966 ± 0.000 |
| `idea_i147_specialist_head_cnn` | 0.851 ± 0.000 | 0.969 ± 0.000 | 0.948 ± 0.000 | 0.894 ± 0.000 | 0.796 ± 0.000 | 0.780 ± 0.000 | 0.826 ± 0.000 | 0.899 ± 0.000 | 0.951 ± 0.000 | 0.977 ± 0.000 |

## Matrix 4 — model × tactic_motif (test PR AUC, restricted to positions tagged with each motif)

| Group | overall | hanging | fork | pin | skewer | overload | discovered_attack | mate_in_1 | promotion | underpromotion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idea_i193_exchange_then_king_dual_stream` | 0.876 ± 0.000 | 0.906 ± 0.000 | 0.855 ± 0.000 | 0.844 ± 0.000 | 0.865 ± 0.000 | 0.853 ± 0.000 | 0.827 ± 0.000 | 0.812 ± 0.000 | 0.652 ± 0.000 | 0.652 ± 0.000 |
| `idea_i024_directed_attack_sheaf_tension_network` | 0.871 ± 0.000 | 0.907 ± 0.000 | 0.849 ± 0.000 | 0.859 ± 0.000 | 0.833 ± 0.000 | 0.841 ± 0.000 | 0.844 ± 0.000 | 0.814 ± 0.000 | 0.596 ± 0.000 | 0.596 ± 0.000 |
| `idea_i087_tactical_radius_filtration` | 0.864 ± 0.000 | 0.902 ± 0.000 | 0.858 ± 0.000 | 0.853 ± 0.000 | 0.832 ± 0.000 | 0.842 ± 0.000 | 0.852 ± 0.000 | 0.802 ± 0.000 | 0.579 ± 0.000 | 0.579 ± 0.000 |
| `idea_i048_rule_automorphism_quotient_bottleneck_network` | 0.861 ± 0.000 | 0.895 ± 0.000 | 0.840 ± 0.000 | 0.824 ± 0.000 | 0.849 ± 0.000 | 0.820 ± 0.000 | 0.825 ± 0.000 | 0.809 ± 0.000 | 0.573 ± 0.000 | 0.573 ± 0.000 |
| `idea_i018_oriented_tactical_sheaf_laplacian` | 0.861 ± 0.000 | 0.900 ± 0.000 | 0.841 ± 0.000 | 0.837 ± 0.000 | 0.830 ± 0.000 | 0.824 ± 0.000 | 0.805 ± 0.000 | 0.764 ± 0.000 | 0.555 ± 0.000 | 0.555 ± 0.000 |
| `idea_i188_tactical_program_induction_network` | 0.861 ± 0.000 | 0.895 ± 0.000 | 0.843 ± 0.000 | 0.815 ± 0.000 | 0.841 ± 0.000 | 0.805 ± 0.000 | 0.813 ± 0.000 | 0.806 ± 0.000 | 0.581 ± 0.000 | 0.581 ± 0.000 |
| `idea_i011_vetoselect_positive_claim_abstention` | 0.858 ± 0.000 | 0.896 ± 0.000 | 0.838 ± 0.000 | 0.818 ± 0.000 | 0.842 ± 0.000 | 0.822 ± 0.000 | 0.828 ± 0.000 | 0.806 ± 0.000 | 0.614 ± 0.000 | 0.614 ± 0.000 |
| `idea_i013_sparse_relation_pursuit_asymmetry` | 0.856 ± 0.000 | 0.896 ± 0.000 | 0.841 ± 0.000 | 0.838 ± 0.000 | 0.815 ± 0.000 | 0.803 ± 0.000 | 0.806 ± 0.000 | 0.817 ± 0.000 | 0.667 ± 0.000 | 0.667 ± 0.000 |
| `idea_i192_latent_reply_entropy_network` | 0.855 ± 0.000 | 0.889 ± 0.000 | 0.838 ± 0.000 | 0.811 ± 0.000 | 0.839 ± 0.000 | 0.826 ± 0.000 | 0.808 ± 0.000 | 0.759 ± 0.000 | 0.548 ± 0.000 | 0.548 ± 0.000 |
| `idea_i191_safe_reply_certificate_verifier` | 0.852 ± 0.000 | 0.889 ± 0.000 | 0.824 ± 0.000 | 0.805 ± 0.000 | 0.827 ± 0.000 | 0.818 ± 0.000 | 0.787 ± 0.000 | 0.767 ± 0.000 | 0.504 ± 0.000 | 0.504 ± 0.000 |
| `idea_i042_legal_automorphism_quotient_network` | 0.852 ± 0.000 | 0.890 ± 0.000 | 0.834 ± 0.000 | 0.809 ± 0.000 | 0.839 ± 0.000 | 0.819 ± 0.000 | 0.820 ± 0.000 | 0.794 ± 0.000 | 0.517 ± 0.000 | 0.517 ± 0.000 |
| `idea_i147_specialist_head_cnn` | 0.851 ± 0.000 | 0.886 ± 0.000 | 0.821 ± 0.000 | 0.798 ± 0.000 | 0.837 ± 0.000 | 0.815 ± 0.000 | 0.795 ± 0.000 | 0.788 ± 0.000 | 0.572 ± 0.000 | 0.572 ± 0.000 |

## Matrix 5 — model × to_move (test PR AUC)

| Group | overall | white | black |
|---|---:|---:|---:|
| `idea_i193_exchange_then_king_dual_stream` | 0.876 ± 0.000 | 0.878 ± 0.000 | 0.873 ± 0.000 |
| `idea_i024_directed_attack_sheaf_tension_network` | 0.871 ± 0.000 | 0.881 ± 0.000 | 0.862 ± 0.000 |
| `idea_i087_tactical_radius_filtration` | 0.864 ± 0.000 | 0.861 ± 0.000 | 0.868 ± 0.000 |
| `idea_i048_rule_automorphism_quotient_bottleneck_network` | 0.861 ± 0.000 | 0.866 ± 0.000 | 0.857 ± 0.000 |
| `idea_i018_oriented_tactical_sheaf_laplacian` | 0.861 ± 0.000 | 0.862 ± 0.000 | 0.860 ± 0.000 |
| `idea_i188_tactical_program_induction_network` | 0.861 ± 0.000 | 0.867 ± 0.000 | 0.856 ± 0.000 |
| `idea_i011_vetoselect_positive_claim_abstention` | 0.858 ± 0.000 | 0.861 ± 0.000 | 0.856 ± 0.000 |
| `idea_i013_sparse_relation_pursuit_asymmetry` | 0.856 ± 0.000 | 0.860 ± 0.000 | 0.852 ± 0.000 |
| `idea_i192_latent_reply_entropy_network` | 0.855 ± 0.000 | 0.853 ± 0.000 | 0.858 ± 0.000 |
| `idea_i191_safe_reply_certificate_verifier` | 0.852 ± 0.000 | 0.855 ± 0.000 | 0.849 ± 0.000 |
| `idea_i042_legal_automorphism_quotient_network` | 0.852 ± 0.000 | 0.856 ± 0.000 | 0.847 ± 0.000 |
| `idea_i147_specialist_head_cnn` | 0.851 ± 0.000 | 0.852 ± 0.000 | 0.851 ± 0.000 |

## Per-slice winners (best 3-seed mean PR AUC for each slice value)

| Slice dim | Slice value | Best group | PR AUC mean ± std | Margin to 2nd |
|---|---|---|---:|---:|
| crtk_difficulty | easy | `idea_i024_directed_attack_sheaf_tension_network` | 0.698 ± 0.000 | +0.037 |
| crtk_difficulty | very_easy | `idea_i024_directed_attack_sheaf_tension_network` | 0.656 ± 0.000 | +0.083 |
| crtk_difficulty | medium | `idea_i193_exchange_then_king_dual_stream` | 0.711 ± 0.000 | +0.008 |
| crtk_difficulty | hard | `idea_i193_exchange_then_king_dual_stream` | 0.792 ± 0.000 | +0.013 |
| crtk_difficulty | very_hard | `idea_i024_directed_attack_sheaf_tension_network` | 0.927 ± 0.000 | +0.001 |
| crtk_phase | opening | `idea_i024_directed_attack_sheaf_tension_network` | 0.864 ± 0.000 | +0.002 |
| crtk_phase | endgame | `idea_i188_tactical_program_induction_network` | 0.854 ± 0.000 | +0.003 |
| crtk_phase | middlegame | `idea_i193_exchange_then_king_dual_stream` | 0.891 ± 0.000 | +0.005 |
| crtk_eval_bucket | winning_black | `idea_i024_directed_attack_sheaf_tension_network` | 0.985 ± 0.000 | +0.004 |
| crtk_eval_bucket | slight_black | `idea_i024_directed_attack_sheaf_tension_network` | 0.859 ± 0.000 | +0.004 |
| crtk_eval_bucket | equal | `idea_i193_exchange_then_king_dual_stream` | 0.817 ± 0.000 | +0.016 |
| crtk_eval_bucket | winning_white | `idea_i033_piece_target_entropic_transport_bottleneck` | 0.975 ± 0.000 | +0.001 |
| crtk_eval_bucket | crushing_black | `idea_i083_fisher_geodesic_tension_network` | 0.979 ± 0.000 | +0.000 |
| crtk_eval_bucket | slight_white | `idea_i024_directed_attack_sheaf_tension_network` | 0.837 ± 0.000 | +0.003 |
| crtk_eval_bucket | crushing_white | `idea_i048_rule_automorphism_quotient_bottleneck_network` | 0.989 ± 0.000 | +0.001 |
| crtk_eval_bucket | clear_black | `idea_i193_exchange_then_king_dual_stream` | 0.917 ± 0.000 | +0.011 |
| crtk_eval_bucket | clear_white | `idea_i087_tactical_radius_filtration` | 0.922 ± 0.000 | +0.001 |
| crtk_to_move | black | `idea_i193_exchange_then_king_dual_stream` | 0.873 ± 0.000 | +0.005 |
| crtk_to_move | white | `idea_i024_directed_attack_sheaf_tension_network` | 0.881 ± 0.000 | +0.003 |
| crtk_tactic_motifs | promotion | `idea_i013_sparse_relation_pursuit_asymmetry` | 0.667 ± 0.000 | +0.014 |
| crtk_tactic_motifs | mate_in_1 | `idea_i013_sparse_relation_pursuit_asymmetry` | 0.817 ± 0.000 | +0.002 |
| crtk_tactic_motifs | hanging | `idea_i024_directed_attack_sheaf_tension_network` | 0.907 ± 0.000 | +0.001 |
| crtk_tactic_motifs | pin | `idea_i024_directed_attack_sheaf_tension_network` | 0.859 ± 0.000 | +0.006 |
| crtk_tactic_motifs | overload | `idea_i193_exchange_then_king_dual_stream` | 0.853 ± 0.000 | +0.011 |
| crtk_tactic_motifs | skewer | `idea_i193_exchange_then_king_dual_stream` | 0.865 ± 0.000 | +0.015 |
| crtk_tactic_motifs | underpromotion | `idea_i013_sparse_relation_pursuit_asymmetry` | 0.667 ± 0.000 | +0.014 |
| crtk_tactic_motifs | fork | `idea_i087_tactical_radius_filtration` | 0.858 ± 0.000 | +0.002 |
| crtk_tactic_motifs | discovered_attack | `idea_i087_tactical_radius_filtration` | 0.852 ± 0.000 | +0.009 |
