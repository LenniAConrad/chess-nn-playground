# PR-AUC reselection report (READ-ONLY, val-only)

**Question:** if best-checkpoint selection had used PR AUC instead of F1, 
which val epoch would have been chosen, and how much higher would *val* PR AUC have been?

**Important caveat:** test PR AUC at the would-be epoch CANNOT be reported here.
Per-epoch checkpoints are not retained, so the only test_pr_auc available is at
the F1-best epoch (i.e., the existing `checkpoint_best.pt`). To get true
corrected test PR AUC, retrain with `training.monitor: pr_auc`.

- results-root: `_combined_view`
- runs analyzed: 241  (skipped 48 without metrics_history.json)
- runs where PR-AUC reselection picks a *different* epoch: 190
- mean val PR AUC lift from reselection: +0.0051
- max  val PR AUC lift: +0.0538

## Top 25 by val PR AUC lift

| run | f1-epoch | pr_auc-epoch | Δepoch | val PR AUC (f1-sel) | val PR AUC (pr-sel) | lift | test PR AUC (f1-sel ckpt only) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `benchmark_bench_residual_medium_simple18_scale_xl_seed44` | 2 | 13 | +11 | 0.8549 | 0.9087 | +0.0538 | 0.8459 |
| `benchmark_bench_residual_small_lc0static112_scale_xl_seed43` | 2 | 15 | +13 | 0.8755 | 0.9117 | +0.0362 | 0.8702 |
| `benchmark_bench_residual_small_lc0static112_seed44` | 3 | 15 | +12 | 0.8789 | 0.9077 | +0.0288 | 0.8747 |
| `benchmark_bench_cnn_small_lc0static112_seed43` | 4 | 13 | +9 | 0.8592 | 0.8875 | +0.0283 | 0.8549 |
| `benchmark_bench_cnn_small_lc0static112_seed42` | 5 | 15 | +10 | 0.8627 | 0.8896 | +0.0270 | 0.8600 |
| `benchmark_bench_lc0_bt4_classifier_scale_up_seed44` | 20 | 14 | -6 | 0.8526 | 0.8778 | +0.0252 | 0.8430 |
| `benchmark_bench_cnn_small_lc0static112_scale_xl_seed42` | 4 | 15 | +11 | 0.8765 | 0.9007 | +0.0242 | 0.8699 |
| `benchmark_bench_cnn_small_simple18_seed43` | 4 | 15 | +11 | 0.8621 | 0.8853 | +0.0232 | 0.8581 |
| `benchmark_bench_cnn_small_lc0static112_scale_xl_seed43` | 5 | 15 | +10 | 0.8783 | 0.9001 | +0.0218 | 0.8737 |
| `benchmark_bench_mlp_simple18_seed44` | 25 | 12 | -13 | 0.6926 | 0.7107 | +0.0180 | 0.6925 |
| `benchmark_bench_residual_small_lc0static112_scale_xl_seed44` | 6 | 14 | +8 | 0.8914 | 0.9091 | +0.0178 | 0.8851 |
| `benchmark_bench_cnn_small_lc0bt4_scale_xl_seed42` | 3 | 13 | +10 | 0.8974 | 0.9135 | +0.0162 | 0.8933 |
| `benchmark_bench_residual_deep_simple18_seed43` | 5 | 12 | +7 | 0.8919 | 0.9080 | +0.0161 | 0.8884 |
| `benchmark_bench_residual_medium_simple18_seed44` | 5 | 12 | +7 | 0.8948 | 0.9108 | +0.0160 | 0.8900 |
| `benchmark_bench_residual_deep_simple18_seed44` | 26 | 11 | -15 | 0.8924 | 0.9084 | +0.0160 | 0.8969 |
| `idea_i011_vetoselect_positive_claim_abstention_scale_xl_seed44` | 13 | 12 | -1 | 0.8608 | 0.8759 | +0.0152 | 0.8460 |
| `benchmark_bench_cnn_small_lc0bt4_scale_up_seed43` | 4 | 14 | +10 | 0.8966 | 0.9113 | +0.0148 | 0.8934 |
| `benchmark_bench_residual_medium_simple18_scale_xl_seed42` | 5 | 14 | +9 | 0.8973 | 0.9110 | +0.0137 | 0.8931 |
| `idea_i009_tactical_equilibrium_network_scale_xl_seed44` | 28 | 13 | -15 | 0.8238 | 0.8375 | +0.0136 | 0.8199 |
| `idea_i011_vetoselect_positive_claim_abstention_seed43` | 18 | 10 | -8 | 0.8613 | 0.8749 | +0.0135 | 0.8507 |
| `benchmark_bench_cnn_small_lc0static112_scale_xl_seed44` | 10 | 19 | +9 | 0.8883 | 0.9014 | +0.0131 | 0.8849 |
| `benchmark_bench_cnn_medium_simple18_seed43` | 8 | 17 | +9 | 0.8876 | 0.9006 | +0.0130 | 0.8837 |
| `idea_i012_dykstra_lcp_seed42` | 16 | 10 | -6 | 0.8436 | 0.8562 | +0.0125 | 0.8328 |
| `idea_i005_null_move_contrast_puzzle_network_scale_xl_seed44` | 25 | 16 | -9 | 0.8299 | 0.8424 | +0.0125 | 0.8204 |
| `benchmark_bench_lc0_bt4_classifier_seed42` | 21 | 12 | -9 | 0.8578 | 0.8703 | +0.0124 | 0.8412 |
