# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/event_delta_bilinear_accumulator.py`.
- Idea-local wrapper: `ideas/registry/p022_event_delta_bilinear_accumulator/model.py`.
- Registry key: `event_delta_bilinear_accumulator`.
- Source primitive: `ideas/research/primitives/external_18_delta_bilinear_ray_blocked_segment_attention.md`
  (rank-1 proposal `primitive_delta_bilinear_accumulator`).

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Per-square tokens are `cat([12 piece planes, side-to-move])`
flattened to `(B, 64, 13)`.

## FM identity vs naive pair enumeration

The naive pair sum `Q = sum_{i<j} (U_i (.) V_j + U_j (.) V_i)` costs
`O(|S|^2 d)`. The closed-form identity

```
A = sum_i U_i
B = sum_i V_i
P = sum_i U_i (.) V_i
Q = A (.) B - P
```

costs `O(|S| d)` and is mathematically exact. The implementation uses
tensor-sum and Hadamard-product operators directly; the constant
factor of 2 from the symmetric expansion is absorbed by the MLP head
weights.

## Static vs incremental update API

At training time the trainer feeds one static board per sample, so
the forward computes `(A, B, P)` from the full active piece set. The
event-update API (`add(u)` / `remove(u)` with O(d) cost per event)
matches the static recompute by construction and is exposed for an
engine inference path. See `math_thesis.md` for the algebraic
guarantee.

## Stop-gradient contract

- `u_proj`, `v_proj`, and the MLP heads receive gradients.
- The occupancy mask flows gradient only through the simple_18 input
  encoding (effectively integer-valued).

## Output dict contract

- `logits` (rebound to `base_logit + primitive_delta`)
- `base_logit`
- `primitive_delta`, `primitive_delta_raw`
- `primitive_gate`, `primitive_gate_logit`, `primitive_gate_entropy`
- `edba_active_count` -- number of occupied squares per sample
- `edba_first_order_magnitude` -- mean |A| per sample
- `edba_pair_term_magnitude` -- mean |Q| per sample

## Ablation modes

See `ablations.md` and `model.ALLOWED_ABLATIONS`. Primary falsifier:
`first_order_only`.

## Deferred internal proposals

Other proposals in the source packet that are *not* implemented here:

- `primitive_ray_blocked_scan` (rank 2) -- covered by p020 and p021.
- `primitive_legal_segment_attention` (rank 3).
- `primitive_exchange_bellman_reducer` (rank 4).
- `primitive_orbit_canonicalizer` (rank 5).

## Why this is not a `ResearchPacketProbe` scaffold

Bespoke `nn.Module` wrapping the bespoke i193 trunk with two
projections, the FM-identity pair term, and head MLPs. Does not
delegate to a shared probe builder.
`implementation_kind: bespoke_model` is consistent with
`audit_implementation_kinds.py`.
