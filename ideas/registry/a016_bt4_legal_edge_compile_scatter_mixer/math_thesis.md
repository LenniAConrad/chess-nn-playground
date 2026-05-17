# Mathematical Thesis — a016 BT4 Primitive Mixer (legal_edge_compile_scatter)

## Scope

This is a controlled architecture study, not a new primitive. The BT4
residual tower is held fixed; only the per-block spatial mixer is swapped.
The mixer under test is the `legal_edge_compile_scatter` adaptation of the
p011 primitive, lifted from the BT4 simple_18 trunk to the generic
`(B, C, 8, 8) -> (B, C, 8, 8)` mixer contract.

## Operator signature

Let `X ∈ R^{B × 64 × C}` be the flattened token tensor and let
`A_r ∈ {0, 1}^{64 × 64}`, `r ∈ {knight, king, rook, bishop, white pawn,
black pawn}`, be the fixed geometric move-pattern adjacencies built in
`_geometric_typed_edges`. Define the per-edge σ-gate and message:

```
g_{r, i, j} = A_r(i, j) · σ( MLP_r( [x_i, x_j] ) )
m_{r, i, j} = W_r · x_i
y_j         = Σ_r  ( 1 / max(eps, Σ_i g_{r, i, j}) ) · Σ_i g_{r, i, j} · m_{r, i, j}
```

LayerNorm on `y`, then `Conv1x1` back to `C` channels, then the BT4 block
applies SqueezeExcite + residual + ReLU.

## Difference from the p011 primitive head

- p011 reads the simple_18 piece planes to compute a *content-conditioned*
  typed legal-move adjacency `A_r(board)`. Here `C` is arbitrary and the
  mixer receives generic features, so the adjacency is degraded to the
  *geometry-only* move-pattern skeleton (constant across batch).
- The σ-gate is therefore the only feature-conditioned term in this
  mixer; in the p011 head the gate was *added on top* of a content-
  conditioned mask.
- All other parts (per-type `W_r`, gate-weighted scatter, typed-degree
  normalisation, LayerNorm) are preserved.

## Claimed advantage

- If the per-edge σ-gate plus typed message-passing structure is
  inherently a better spatial mixer than the lc0_bt4 conv pair, this
  architecture should beat `mixer: conv` under an identical tower,
  optimizer, and data contract.
- Falsifier: if `mixer: conv` matches or beats this mixer on
  puzzle_binary aggregate plus the tactical CRTK slices, the primitive
  is not load-bearing as a spatial mixer at this scale.

## What is actually proven

- Mixer build, forward shape `(B, C, 8, 8) -> (B, C, 8, 8)`, and a
  backward smoke test under the gating script in
  `scripts/ideas/scaffold_bt4_primitive_mixers.py`.

## What is hypothesised

- Slice-level lift over the `mixer: conv` and `mixer: attention`
  baselines on the CRTK tactical and high-difficulty splits. Not
  measured until `CLAUDE_ALLOW_TRAINING=1`.

## Failure cases

- The dense `(B, 64, 64, 2C)` concat tensor materialised by the edge-
  gate MLPs is the dominant memory term and scales as `C^2`. Larger
  `channels` or `num_blocks` may OOM before reaching paper-grade
  reliability tiers.
- The geometric-only adjacency removes the content-conditioned masking
  that made p011 a chess-aware operator. Lift over conv is now an
  empirical question, not a structural guarantee.
