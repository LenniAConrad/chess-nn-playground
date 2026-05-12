# Ablations

- `model.ablation: cnn_only_matched` zeros the variational branch inputs to the head
  while retaining the board CNN path and classifier shape.
- `model.ablation: action_only` keeps action scalar and energy summaries but removes
  residual summaries and residual-map features from the classifier input.
- `model.ablation: no_gradient_terms` removes `Dx u` and `Dy u`, leaving potential-force
  terms only.
- `model.ablation: random_difference_operators` replaces board finite differences with
  fixed random local operators of the same support and output shape.
- `model.ablation: residual_norm_only` keeps scalar residual summaries but removes the
  residual-map CNN.
- `model.ablation: force_head_only` predicts the residual directly from context and
  skips the variational divergence term.
- `model.ablation: harmonic_control` uses a fixed Laplacian-style residual instead of
  the learned stiffness action.
