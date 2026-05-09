# Architecture

`Defender-Exhaustion Cascade Network` is a bespoke implementation of
idea `i178`. It models a chess puzzle as a small recurrent
allocation game: typed *obligation* tokens compete for a finite pool
of typed *resource* tokens across `cascade_steps`, and at each step
allocations consume resource capacity. The puzzle head reads the
exhaustion curve (per-step residual demand, allocation entropy, and
max deficit) plus pooled board context and returns one logit.

## Pipeline

- Input: board tensor `(B, 18, 8, 8)`. CRTK / source metadata is
  reporting-only and never used as model input.
- A compact convolutional trunk lifts each square to `channels`
  features.
- A `1 x 1` projection produces per-square `token_dim` features. Two
  banks of learnable per-type spatial queries —
  `obligation_type_queries` of size `(obligation_types, token_dim)`
  and `resource_type_queries` of size `(resource_types, token_dim)` —
  perform softmax pooling over the 64 squares to build typed
  *obligation tokens* and *resource tokens*. Type-identity
  embeddings are added on top.
- Each resource token gets a scalar capacity in `(0, capacity_init]`
  via a small MLP and a `sigmoid * capacity_init` gate.
- Pooled trunk features (mean and max) are projected to a
  `token_dim` *threat context* vector that drives the cascade.
- The compatibility matrix `compat = obl @ res^T / (sqrt(D) *
  allocation_temperature)` of shape `(B, obligation_types,
  resource_types)` is the static "this resource can serve this
  obligation" signal.
- A GRU cell `cascade_cell` of hidden size `token_dim` updates a
  per-obligation `demand_state` from the threat context. This is the
  "obligation_update" step from the source packet.
- At each cascade step `t`:
  - `demand_state_t = GRU(threat_context, demand_state_{t-1})`
  - `demand_t = softplus(demand_head(demand_state_t))` — scalar need
    per obligation.
  - `modulator_t = demand_modulator(demand_state_t)` — per-`(obl,
    res)` bias that pushes allocation away from obligation-resource
    pairs the obligation is currently stressing.
  - `pressure_t` is the accumulated allocation × capacity from earlier
    steps; it grows monotonically and *exhausts* resource availability.
  - `allocation_t = softmax_j(compat[i, j] - demand_pressure *
    (pressure_t[i, j] + modulator_t[i, j]))` — a per-obligation
    softmax over resources.
  - `allocated_t[i] = sum_j allocation_t[i, j] * capacity[j]`.
  - `residual_t[i] = demand_t[i] - allocated_t[i]`.
  - Curve diagnostics: `sum_i softplus(residual_t)`,
    `mean_i entropy(allocation_t[i, :])`, and
    `max_i softplus(residual_t)`.
  - `pressure_{t+1} = pressure_t + allocation_t * capacity`.
- A LayerNorm + GELU MLP head consumes pooled trunk features (mean
  and max), the per-step exhaustion curves, the final per-obligation
  residual, the final per-resource allocation marginal, and the final
  scalar demand / allocated / residual totals, and returns one puzzle
  logit.

## Distinction From Hall-Defect / Matching Ideas

Hall-defect and matroid ideas in this repository (e.g. i035
`hall_defect_obligation_matroid`, i083 `hall_defect_zeta_operator`)
read a one-shot Hall statistic off a static set system. This
architecture is a *recurrent* cascade: resource capacity is consumed
at every step, allocation pressure compounds through `pressure_t`,
and the puzzle signal lives in the *trajectory* of residual demand
across steps, not in a single matching deficit. Setting
`cascade_steps = 1` recovers a much weaker static-allocation
baseline.

## Implementation Binding

- Registered model name: `defender_exhaustion_cascade_network`
  (registered in `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/defender_exhaustion_cascade_network.py`
  (`DefenderExhaustionCascadeNetwork` and
  `build_defender_exhaustion_cascade_network_from_config`).
- Idea-local wrapper:
  `ideas/i178_defender_exhaustion_cascade_network/model.py` calls
  `build_defender_exhaustion_cascade_network_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this
  idea.
