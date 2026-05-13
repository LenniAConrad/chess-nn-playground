# Math Thesis

Source: `ideas/research/primitives/external_26_delta_update_occlusion_ray_piece_kernels.md`
(Occlusion-Aware Ray Scan, OARS — first-ranked proposal).

## Operator

Let `X in R^{B x F x 8 x 8}` be a per-square feature stack. For each
chess direction `d in {N, NE, E, SE, S, SW, W, NW}`, OARS runs a
selective associative scan whose state is the running sum and whose
blocker decision depends on the *intermediate state*:

```
state_{i, d}      = features_i + sigma(W_block_d * state_{i-d, d}) * state_{i-d, d}
y_i               = sum_d C_d * state_{i, d}
```

The associative operator is `a (x) b = a + sigma(W_block(a)) * b`. In
the implementation we materialise this iteratively for
`max_ray_length` steps; each step shifts the running state by one
square along the direction and updates the gate from the per-(square,
direction) sigmoid of a learned linear head on the *raw* per-square
features `features_i` (a fixed-feature simplification of the
state-dependent gate that keeps the scan stable on the small 8x8
board).

## What is proven

- Without the blocker gate the operator collapses to a plain
  geometric prefix sum (the `disable_blocker_gate` ablation), so
  comparing the unablated and ablated runs isolates the selective-
  gating contribution.
- The scan is bounded for sigmoid gates because `sigma in (0, 1)`
  contracts the previous state at each step.
- The implementation uses the same `_shift_along_direction` helper
  as `p026`, so the per-direction geometry is consistent across the
  ray-family primitives.

## What is hypothesised

- The blocker gate learns to terminate a ray on *features* (e.g.
  "first hostile piece", "specific piece type") rather than just
  raw occupancy. This is the differentiator from `p026` RayPool.
- The state-dependent gating lets a single OARS pass express
  long-range interactions that the trunk's stacked convs would
  otherwise need multiple layers to discover.

## Architecture-level claim

```
final_logit(x) = i193_trunk(x) + primitive_gate(x) * primitive_delta(x)
```

The OARS pooled output is fused with the trunk diagnostics into the
gate / delta MLPs.

## Failure cases

- If `sigma(W_block)` collapses to 0 everywhere, the head reduces to a
  per-square 1x1 conv followed by a mean pool — a trivial baseline.
- If `sigma(W_block)` collapses to 1 everywhere, the scan is a plain
  geometric prefix sum and the head reduces to a variant of `p026` /
  RayPool without occupancy-based termination.
- The sequential loop is small on 8x8 but would benefit from a fused
  CUDA kernel for larger grids.

## Falsifiers

- `disable_blocker_gate`: force the blocker gate to 1.0. Tests
  whether the selective gating is load-bearing.
- `shuffle_directions`: random permutation of the 8 directions.
  Decouples direction-specific learning.
- `zero_oars_features`: replace the pooled output with zeros. Tests
  the trunk-diagnostics contribution to the fusion vector.
