# Math Thesis

Source: `ideas/research/primitives/external_34_active_esp_conflict_matching_eikonal_primitives.md`,
rank-3 proposal `primitive_occupancy_eikonal`. The rank-1 proposal in
the same packet (`primitive_active_esp`) duplicates `p024
event_symmetric_interaction_accumulator` and is not re-implemented.

## Working thesis

For the fixed 8x8 king-neighbour graph `G = (V, E)` with vertex set
`V = {0..63}` and per-vertex eight-neighbourhood edges, learned edge
costs `c >= 0`, and learned seed costs `s >= 0`:

    T_v = softmin_tau( s_v , { T_u + c_{uv} : (u, v) in E } ),
    softmin_tau(a) = -tau * log sum_i exp(-a_i / tau).

The fixed point is reached by relaxation: starting from `T^{(0)} = s`,
iterate

    T^{(k+1)}_v = softmin_tau( s_v , { T^{(k)}_u + c_{uv} : (u, v) in E } ),

for `k = 0..num_iterations - 1`. Gradients flow through every
relaxation step.

## Boundary handling

Out-of-board "neighbours" are represented as self-loops in the
`neighbours` buffer. The candidate `T_v + c_v` is always at least as
large as `T_v`, so the self-loop is harmless under softmin. This keeps
the gather contiguous and avoids per-cell masking.

## Edge-cost contract

Per-channel edge costs `c >= 0` are required by the soft-min eikonal
operator. We project the trunk joint feature through softplus and add
`cost_bias >= 0` so the cost field is strictly positive even when the
projection learns small magnitudes. We use a *node-cost* convention --
`c_{uv} = c_v` -- which is standard in fast-marching methods. Replacing
this with separate per-edge costs would mean projecting eight times as
many parameters; deferred to future work if the falsifier passes.

## Architecture-level claim

    final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * delta(field_mean, field_max, field_min)

`field_{mean, max, min} in R^{B, q_channels}`, pooled over the 64
squares. The gate is initialised closed (`gate_init = -2.0`).

## Falsifiers

- Primitive-level: `shuffle_field` (in-batch permutation of the arrival
  field) must lose the slice lift.
- `single_iteration` (cap relaxation at one step) must lose at least a
  portion of the lift -- if it matches the full operator, propagation
  beyond the immediate neighbourhood is not load-bearing.
- `uniform_costs` (constant edge costs) must lose the cost-conditional
  component.
- Architecture-level: p039 must beat i193 on its declared slice (king
  safety / escape-corridor positions) without regressing aggregate PR
  AUC.

## Why this is not Conv2d

A 3x3 Conv2d integrates local information *additively* with a learned
kernel; the eikonal transform integrates it through a `min-plus`
fixed-point, which propagates globally with bounded iteration count.
The active predecessor set is input-dependent and changes during
relaxation -- a property no fixed-stencil convolution shares.

## Why this is not message passing

Standard message passing aggregates by sum / mean / max with a *learned*
aggregator. The eikonal soft-min uses a fixed `(a, b) -> -tau * log(exp(-a/tau) + exp(-b/tau))`
aggregator that approximates the tropical / min-plus semiring. The
gradient through this aggregator is the standard soft-min Jacobian (a
Gibbs distribution over the candidates), not a learned per-edge weight.
