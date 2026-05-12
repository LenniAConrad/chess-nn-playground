# Implementation Notes

- Central code: `src/chess_nn_playground/models/polar_procrustes_alignment_bottleneck.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i063_polar_procrustes_alignment_bottleneck/model.py` (delegates to `build_polar_procrustes_alignment_bottleneck_from_config`).
- Registry key: `polar_procrustes_alignment_bottleneck`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2104_friday_shanghai_polar_procrustes.md`.
- Board input only (simple_18, 18 planes); engine, source, verification, and CRTK metadata are never used as input.
- Central operator: batched `torch.linalg.svd` of the cross-covariance `C(x) = X(x)^T Y(x)` with explicit polar / Procrustes recovery (`Q* = U V^T`, `H = V Sigma V^T`).
- Default config uses `matrix_space: embedding`, `token_dim: 48`, `role_count: 8`, with a `cross_cov_eps * diag(1, 2, ..., M) / M` tilt that keeps SVD backward stable when singular values would otherwise coincide.
- Section 9 falsifiers exposed via `model.ablation`: `separate_matrix_stats_only`, `identity_alignment_only`, `random_orthogonal_alignment`, `batch_shuffled_opponent`, `material_only_matrices`, `role_pool_mean_only`, `singular_values_only`.
