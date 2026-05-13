# Math Thesis

Source: `ideas/research/primitives/external_27_ray_parallel_ssm_delta_accumulator_sparse_conv.md`
(Ray-Parallel Selective State Space Model, Ray-SSM — first-ranked proposal).

## Operator

Let `x in R^{B x F x 8 x 8}` be a per-square feature stack. For each
chess direction `d in {N, NE, E, SE, S, SW, W, NW}`, Ray-SSM runs a
selective state-space scan with diagonal A/B and a learned read-out
C:

```
h_{i, d, c} = A_{i, d, c} * h_{i - shift_d, d, c} + B_{i, d, c} * x_{i, c}
y_{i, c}   = sum_d C_{d, c} * h_{i, d, c}
```

with `A_{i, d, c} = sigma(W_A(x_i))_{d, c}` and
`B_{i, d, c} = sigma(W_B(x_i))_{d, c}` — both input-conditioned per
(direction, channel) in (0, 1). `C` is a learned parameter of shape
`(NUM_DIRECTIONS, F)`.

## What is proven

- Diagonal A and B keep the recurrence stable for sigmoid inputs in
  (0, 1) because both factors are contractions.
- The scan structure is associative inside each direction (a special
  case of the Mamba selective scan); we use the sequential
  implementation for clarity but a parallel-scan kernel would emit
  the same gradient.
- With `A = 0`, the head collapses to a per-square `B * x` followed
  by mean pool — a trivial baseline (covered by the
  `disable_selective_A` ablation).
- With `B = 0`, the state never gets injected and the head decays to
  zero (covered by the `disable_selective_B` ablation).

## What is hypothesised

- Selective state-space updates with separately-learned A and B let
  the head learn a *mixture* of retention vs. injection along the
  ray. RayPool's geometric prefix sum has only one parameter
  (`gamma`), and OARS's blocker gate is multiplicative-only;
  Ray-SSM is the strictly more expressive operator in this family.
- Long-range tactics benefit from per-channel mixing along the ray
  (e.g. "carry the attacker's piece-value but reset the defender
  count").

## Architecture-level claim

```
final_logit(x) = i193_trunk(x) + primitive_gate(x) * primitive_delta(x)
```

The pooled SSM output is fused with the trunk diagnostics into the
gate / delta MLPs.

## Failure cases

- The trunk's bishop/rook/queen attack masks plus the `p026` RayPool
  primitive may collectively already saturate the ray-aware signal,
  in which case Ray-SSM adds no marginal value.
- The sigmoid A is bounded above by 1; a long enough ray will
  attenuate distant features. This is a feature, not a bug, but
  limits the effective receptive field at small sigmoid values.
- The C parameter is per-direction-only (not per-square); the spec's
  full form `y = sum_d C_{i, d} h_{i, d}` would require an additional
  parameter table conditioned on square. We keep the simpler form
  here and document the deferred extension in `implementation_notes.md`.

## Falsifiers

- `disable_selective_A`: force A to a constant. Tests whether the
  input-conditioned retention is load-bearing.
- `disable_selective_B`: force B to a constant. Tests whether the
  input-conditioned injection is load-bearing.
- `no_directional_C`: replace each per-direction C with the mean
  across directions. Tests whether direction-specific read-out is
  load-bearing.
- `zero_ssm_features`: zero the pooled SSM output. Tests the trunk
  diagnostics' contribution to the fusion vector.
