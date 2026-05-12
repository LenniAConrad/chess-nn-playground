# Ablations

The bespoke model exposes the following ablation switch via
`model.ablation` in the config:

- `none` (default) — full source-invariant bottleneck: shared trunk over the
  symmetry orbit (identity, file flip, rank flip, 180-rotation), shared
  bottleneck MLP, orbit-mean code with learned residual-direction
  subtraction, puzzle head + aux residual head.
- `no_invariance` — drop the symmetry orbit and use only the identity view.
  Tests whether multi-view averaging matters.
- `no_orthogonalization` — keep the orbit but skip the explicit
  residual-direction subtraction (`c_main = c̄`). Tests whether mean-pooling
  alone is sufficient.
- `no_aux_residual_logit` — disable the auxiliary residual logit head.
  Tests whether the residual-channel diagnostic is load-bearing.

External baselines for the same `puzzle_binary` benchmark contract:

- LC0 BT4 (`model.name: lc0_bt4_classifier`)
- NNUE (`model.name: nnue`)
- The strongest registered idea runs on the same split and seeds.
