# Mathematical Thesis — a014 BT4 Primitive Mixer (legal_move_graph_delta)

This is a controlled architecture study, not a new primitive. The
mathematical claims about the spatial mixer itself live in
`ideas/registry/p009_legal_move_graph_delta/math_thesis.md`. The thesis
here is about the *swap*: holding a BT4-style residual tower, optimizer
protocol, and data contract fixed, and replacing only the per-block
spatial mixer with the `legal_move_graph_delta` primitive adapted to a
shape-preserving mixer contract.

## Operator signature

The BT4 residual block factors as

```
y = ReLU(SqueezeExcite(M(x)) + x)
```

where `x, y ∈ R^{B × C × 8 × 8}`. The shared `bt4_primitive_mixer`
tower fixes the stem, residual depth `N`, SqueezeExcite reduction, and
value head; the only operator that varies between sibling
`a###_bt4_*_mixer` ideas is the mixer `M`.

In `a014` we instantiate

```
M(x) = legal_move_graph_delta_mixer(x)
```

adapted from the `(B, 64, d)` token-tensor primitive form to the
`(B, C, 8, 8) -> (B, C, 8, 8)` mixer contract by flattening the spatial
axes, applying the per-type message passing along the legal-move
adjacency `A_r(X_b) ∈ {0,1}^{64×64}` for each piece type `r`, and
reshaping back. The typed adjacency is computed inside
`torch.no_grad()` from the same `simple_18` board the trainer already
consumes.

## Claimed advantage of the swap

Because the only changed term in the residual recurrence is `M`, any
difference in puzzle_binary metrics between `a014` and the sibling
`bt4_conv_mixer` / `bt4_attention_mixer` baselines is attributable to
the mixer choice. A measurable lift would mean the chess-aware
`legal_move_graph_delta` operator beats both conv and dense attention
as a per-block spatial mixer at this tower depth and width.

## Assumptions

- The BT4 tower (`bt4_primitive_mixer`) is held byte-for-byte fixed
  across the `a###_bt4_*_mixer` sweep.
- Optimizer, loss, batch size, augmentation, scheduler, and seed are
  the same across the sweep (see `config.yaml`).
- The mixer obeys the shape-preserving contract — input
  `(B, C, 8, 8)`, output `(B, C, 8, 8)`, no per-block parameter sharing
  outside the mixer itself.

## What is actually proven

- The model builds, runs a forward and backward pass under the shared
  `bt4_primitive_mixer` registration (smoke-tested before this folder
  was created — see `idea.yaml notes`).
- The wrapper enforces the `mixer = legal_move_graph_delta` contract
  via `build_model_from_config`.

## What is only hypothesised

- Whether the swap actually improves puzzle_binary aggregate or
  CRTK-sliced metrics versus the conv / attention baselines.
- Whether any lift survives the `random_typed_edges` / `shared_weight`
  / `no_normalization` falsifiers documented in the source primitive's
  `ablations.md`.

## Failure cases

- The typed-adjacency construction dominates wall-clock at this tower
  width and the run cannot complete inside the shared training budget.
- The per-type linears under-utilise their parameter budget at small
  channel widths, producing noisy gradients that the residual sum
  cannot dampen.
- The mixer's lift, if any, fails to survive the
  `random_typed_edges` ablation, in which case the architecture is
  no better than dense attention with the same parameter budget.
