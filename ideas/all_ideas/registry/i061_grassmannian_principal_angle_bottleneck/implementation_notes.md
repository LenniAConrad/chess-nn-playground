# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/grassmannian_principal_angle_bottleneck.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i061_grassmannian_principal_angle_bottleneck/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_grassmannian_principal_angle_bottleneck_from_config`.
- Registry key: `grassmannian_principal_angle_bottleneck`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2058_friday_shanghai_grassmannian_angles.md`.
- Input is the simple_18 board tensor only; CRTK / source / engine /
  verification metadata are never consumed by the model.
- The covariance matrix is regularised by `covariance_eps * I_D`
  (default `1e-3`) before `torch.linalg.eigh`, which keeps the
  eigendecomposition well-conditioned even when fewer than `subspace_dim`
  occupied tokens contribute to a role.
- Default geometry: `token_dim = 48`, `role_count = 8`,
  `subspace_dim = 6`, giving `28` unordered role-pairs and a
  per-pair spectrum of `K = 6` cosines (and matching angles).
- The `ablation` field selects between `none`, `no_cross_angles`,
  `batch_shuffled_angles`, `eigenvalues_only`, `pooled_token_head`, and
  `no_orthonormalization`. All ablations preserve the head input
  dimensionality so capacity is matched.
- Some GPU eigensolvers may have nondeterministic kernels; CPU smoke
  tests remain deterministic. The forward pass is otherwise seed-stable.
