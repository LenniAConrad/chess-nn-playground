# Mathematical Thesis — a013 BT4 Primitive Mixer (rule_conditioned_sparse_attention)

This is a controlled architecture study, not a new primitive. The
mathematical claims about the spatial mixer itself live in
`ideas/registry/p008_rule_conditioned_sparse_attention/math_thesis.md`.
The thesis here is about the *swap*: holding a BT4-style residual tower,
optimizer protocol, and data contract fixed, and replacing only the
per-block spatial mixer with the `rule_conditioned_sparse_attention`
primitive adapted to a shape-preserving mixer contract.

## Operator signature

The BT4 residual block factors as

```
y = ReLU(SqueezeExcite(M(x)) + x)
```

where `x, y ∈ R^{B × C × 8 × 8}`. The shared `bt4_primitive_mixer`
tower fixes the stem, residual depth `N`, SqueezeExcite reduction, and
value head; the only operator that varies between sibling
`a###_bt4_*_mixer` ideas is the mixer `M`.

In `a013` we instantiate

```
M(x) = rule_conditioned_sparse_attention_mixer(x)
```

adapted from the `(B, 64, d)` token-tensor primitive form to the
`(B, C, 8, 8) -> (B, C, 8, 8)` mixer contract by flattening the spatial
axes, applying the mixer, and reshaping back. The legal-move adjacency
`A(X_b) ∈ {0,1}^{64×64}` is computed inside `torch.no_grad()` from the
same `simple_18` board the trainer already consumes.

## Claimed advantage of the swap

Because the only changed term in the residual recurrence is `M`, any
difference in puzzle_binary metrics between `a013` and the sibling
`bt4_conv_mixer` / `bt4_attention_mixer` baselines is attributable to
the mixer choice. A measurable lift would mean the chess-aware
`rule_conditioned_sparse_attention` operator beats both conv and dense
attention as a per-block spatial mixer at this tower depth and width.

## Assumptions

- The BT4 tower (`bt4_primitive_mixer`) is held byte-for-byte fixed
  across the `a###_bt4_*_mixer` sweep.
- Optimizer, loss, batch size, augmentation, scheduler, and seed are
  the same across the sweep (see `config.yaml`).
- The mixer obeys the shape-preserving contract — input
  `(B, C, 8, 8)`, output `(B, C, 8, 8)`, no per-block parameter sharing
  outside the mixer itself.

## What is actually proven

- The model builds, runs a forward and backward pass under the
  shared `bt4_primitive_mixer` registration (smoke-tested before this
  folder was created — see `idea.yaml notes`).
- The wrapper enforces the `mixer = rule_conditioned_sparse_attention`
  contract via `build_model_from_config`.

## What is only hypothesised

- Whether the swap actually improves puzzle_binary aggregate or
  CRTK-sliced metrics versus the conv / attention baselines.
- Whether any lift survives the `random_edges` / `untied_state` /
  `single_iteration` falsifiers documented in the source primitive's
  `ablations.md`.

## Failure cases

- The mixer dominates compute or memory and the run cannot complete
  inside the shared training budget.
- Numerical instability in the selective recurrence on positions with
  very heterogeneous mobility, propagating into the residual sum
  before SqueezeExcite can dampen it.
- The mixer's lift, if any, fails to survive a randomised-adjacency
  ablation, in which case the architecture is no better than dense
  attention with the same parameter budget.
