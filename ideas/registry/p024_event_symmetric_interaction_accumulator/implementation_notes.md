# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/event_symmetric_interaction_accumulator.py`.
- Idea-local wrapper: `ideas/registry/p024_event_symmetric_interaction_accumulator/model.py`.
- Registry key: `event_symmetric_interaction_accumulator`.
- Source primitive: `ideas/research/primitives/external_20_event_symmetric_sparse_scatter_ray_scan.md`
  (rank-1 proposal `primitive_event_symmetric_accumulator`).

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Per-square tokens are `cat([12 piece planes, side-to-move])`
flattened to `(B, 64, 13)`, then projected to a `token_dim`-d vector
per square.

## Streaming recurrence

The recurrence walks over all 64 squares once per sample. For each
square, the order updates run from `R` down to `1`:

```python
for s in range(64):
    u = tokens[:, s, :] * occupancy[:, s, None]
    for r in range(order, 0, -1):
        if r == 1:
            E[0] += u
        else:
            E[r - 1] += u * E[r - 2]
```

For `R = 2`, the result is mathematically identical to the
factorisation-machine identity:

```
E^{(2)} = (1/2) * (S1 (.) S1 - sum_i u_i (.) u_i),    S1 = sum_i u_i
```

The streaming recurrence generalises to `R = 3` without enumerating
triples. (The closed-form for `R = 3` exists but involves third
power sums that need careful normalisation; the streaming recurrence
is the simpler and more numerically stable option.)

## Static vs incremental update API

At training time the trainer feeds one static board per sample, so
the streaming recurrence runs over the full active piece set. The
`add(u)` / `remove(u)` API is preserved as documentation: the
forward recurrence is exactly `add` applied over the full set, and
`remove(u)` is its exact inverse with the same `O(R d)` cost. Engine
inference would use the incremental path; training does not.

## Stop-gradient contract

- `token_proj` and the MLP heads receive gradients.
- The occupancy mask is rule-derived from the simple_18 piece planes.

## Output dict contract

- `logits`, `base_logit`, `primitive_delta`, `primitive_delta_raw`
- `primitive_gate`, `primitive_gate_logit`, `primitive_gate_entropy`
- `esia_active_count` -- number of occupied squares per sample.
- `esia_order_max_magnitude`, `esia_order_mean_magnitude`.
- `esia_order_<r>_magnitude` for each order `r = 1..R`.

## Ablation modes

See `ablations.md` and `model.ALLOWED_ABLATIONS`. Primary falsifier:
`first_order_only`.

## Deferred internal proposals

Other proposals in the source packet that are *not* implemented here:

- `primitive_rule_generated_sparse_scatter` (rank 2).
- `primitive_first_blocker_ray_scan` (rank 3) -- covered by p020 / p021.
- `primitive_chess_irrep_orbit_norm` (rank 4).
- `primitive_counterfactual_delta_map` (rank 5) -- partially related to
  the i246 promotion-aware head, which already implements a
  counterfactual fanout primitive.

## Why this is not a `ResearchPacketProbe` scaffold

Bespoke `nn.Module` wrapping the bespoke i193 trunk with explicit
Python-loop streaming recurrence over orders and squares. Does not
delegate to a shared probe builder.
`implementation_kind: bespoke_model` is consistent with
`audit_implementation_kinds.py`.
