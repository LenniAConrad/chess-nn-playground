# Mathematical Thesis — p010 Ray-Occlusion Semiring Scan

## Operator signature

Let `X ∈ R^{B × 64 × d}` be per-square token embeddings, `O_b ∈ [0, 1]^{64}`
be the occupancy, `π_δ(s, k)` the target square at step `k` of ray
direction `δ`, and `λ_δ ∈ [0, 1]` a learned per-direction decay scalar.
Define ray transmittance

```
T_{b, s, δ, k} = Π_{u < k} (1 - O_{b, π_δ(s, u)})
```

(probability the ray launched from `s` in direction `δ` reaches step
`k` unblocked). Computed in log-domain prefix sums with `(1 - O)`
clamped away from 0 for numerical stability.

For per-direction linear maps `W_δ` (or one shared linear under
`constant_direction`):

```
y_{b, s, δ} = W_δ · Σ_{k=1..L_{s, δ}} T_{b, s, δ, k} · λ_δ^k · X_{b, π_δ(s, k)}
```

The 8 per-direction outputs are concatenated to `(B, 64, 8 · d_ray)`,
mean-pooled over squares, and projected by an MLP to the scalar delta.

```
final_logit = base_logit + σ(W_gate · trunk_pool) · MLP(mean_pool(y))
```

## Why this is not just depthwise conv or attention

- Convolution has fixed local kernels and cannot express the multi-
  plicative "stop when blocked" visibility.
- Attention with an `attn_mask` materialises an n^2 logit matrix and
  uses softmax; this operator never instantiates an n^2 score map and
  the weights are deterministic prefix products on an occlusion
  semiring rather than learned softmax scores.
- Closest peer is Mamba's selective scan, but Mamba runs along a 1-D
  sequence with input-conditioned `A, B, C, Δ`; here the topology is
  multi-ray geometric and the per-step weight is the occupancy-derived
  transmittance, not an input-conditioned learned scan parameter.

## Claimed advantage

- Sliding-piece tactics (pins, skewers, x-rays, discovered attacks)
  are first-order ray problems: the relevant signal is "first blocker
  on this ray". The semiring scan delivers exactly that geometry.
- Cost is `O(B · 8 · 64 · 7 · d)` plus 8 per-direction linears — a
  small constant factor over depthwise conv.

## What is actually proven

- Transmittance values are clamped, finite, and bounded in [0, 1].
- The unit tests verify that the scan returns finite values for the
  starting position and for a tactical FEN.
- Backward through the scan is gradient-friendly because the log-
  prefix-product is differentiable.

## What is hypothesised

- The "FP-rate drop ≥ 5%" target from the file's spec is **not**
  measured. Scout training is gated by `CLAUDE_ALLOW_TRAINING=1`.
- `uniform_transmittance` tests whether occlusion is load-bearing;
  `no_step_decay` tests whether step-decay is load-bearing.

## Failure cases

- Long products vanish for highly-occupied positions. Log-domain
  prefix products with the `(1 - O).clamp(eps, 1)` floor avoid
  numerical zero but can still saturate the transmittance gradient.
- Off-board ray entries are masked to zero via `ray_step_valid`, so
  the cumulative product never "leaks" past board boundaries.
