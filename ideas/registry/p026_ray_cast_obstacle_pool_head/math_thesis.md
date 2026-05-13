# Math Thesis

Source: `ideas/research/primitives/external_22_ray_cast_obstacle_pooling_sparse_emit.md`
(Ray-Cast Obstacle Pooling, RayPool — first-ranked proposal).

## Operator

Let `X in R^{B x F x 8 x 8}` be a per-square feature stack and
`O in [0, 1]^{B x 8 x 8}` the board occupancy. For each direction
`d in {N, NE, E, SE, S, SW, W, NW}` with offset `(dr_d, df_d)` and a
learned per-direction decay `gamma_d in [0, 1]`, RayPool computes

```
Y_{d, i} = sum_{s>=1} gamma_d^s * X_{i + s * (dr_d, df_d)}
                       * prod_{k=1..s-1} (1 - O_{i + k * (dr_d, df_d)})
```

with cells outside the 8x8 grid contributing zero. The output is
`(B, 8, F, 8, 8)` of per-direction pooled features.

## What is proven

- The geometric series terminates at the first occupied square along the
  ray because the running product of `(1 - O)` collapses to 0 there.
- The operator is associative along each direction, so we can implement
  it as a sequential prefix sum of length `max_ray_length` per direction.
- The decay `gamma_d` ensures bounded magnitude: `||Y_{d, i}|| <=
  ||X||_inf * gamma_d / (1 - gamma_d)` for unbounded rays; for the 8x8
  board the partial sum is always bounded.

## What is hypothesised

- Long-range piece influence — back-rank pressure, pins, batteries — is
  more directly representable as a per-ray pooled signal than as the
  layered receptive field of a 3x3 conv stack.
- Adding this signal as a side head on top of the i193 trunk improves the
  puzzle logit on slices where the tactical motif is long-range and the
  trunk's exchange/king features under-represent the ray.
- The learned per-direction decay `gamma_d` lets the head specialise:
  pawns/king moves prefer short rays, sliders prefer long rays.

## Architecture-level claim

```
final_logit(x) = i193_trunk(x) + primitive_gate(x) * primitive_delta(x)
```

where `primitive_delta` reads the LayerNorm of the mean-pooled
`(NUM_DIRECTIONS * feature_dim)` flat vector and combines it with the
trunk diagnostics.

## Failure cases

- The trunk's bishop/rook/queen attack masks already encode unblocked ray
  influence. RayPool may add no marginal signal in that regime; the gate
  collapses to zero and the head is silent.
- Floating-point drift along long rays at high gamma is possible but
  bounded; we cap `max_ray_length` at 7 (the longest chess ray) and
  clamp `gamma_d` to `[0, 1]`.
- The implementation is sequential per direction. On an 8x8 board this is
  negligible, but it would not scale to larger grids without a fused
  CUDA kernel.

## Falsifiers

- `drop_occlusion`: ignore the blocker mask. If the unablated and
  ablated runs match on long-range slices, occlusion termination is not
  load-bearing.
- `shuffle_directions`: random per-pass permutation of the 8 directions
  (the parameter `gamma` gets shuffled too). If the shuffled run
  matches, direction-specific learning is not load-bearing.
- `zero_rays`: replace the pooled feature vector with zeros; tests
  whether the trunk diagnostics in the fusion vector are doing all the
  work.
