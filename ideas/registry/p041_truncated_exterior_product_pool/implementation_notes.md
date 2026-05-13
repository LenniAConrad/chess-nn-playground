# Implementation Notes

## Worktree

Implemented in the `cnp-primitive-gpt-orbit-boolean-algebra` worktree.

## Module

`src/chess_nn_playground/models/primitives/truncated_exterior_product_pool.py`

Contains:

- `TruncatedExteriorProductPool` nn.Module.
- `build_truncated_exterior_product_pool_from_config`.
- `_piece_tokens`, `_grade_indices` helpers.
- `_build_extension_table` for the per-grade scatter target / sign
  tables.
- `ALLOWED_ABLATIONS` tuple.

## Why the wedge tables are precomputed at construction time

The wedge update `M' = M * (1 + z_i)` is essentially a sparse linear
operation in the basis of `Lambda^k(R^r)`. The sparsity pattern (which
basis index each contribution lands in) and the signs (from the
permutation parity) depend only on `r` and `k` -- not on the input
data. Precomputing these as buffers at construction time means the
per-token forward is a small batch of `scatter_add_` calls, with no
Python branching inside the inner loop.

## Why r and R must stay small

`D_R = sum_k C(r, k)` blows up quickly: for `r = 4, R = 3` it is 15,
for `r = 6, R = 3` it is 42, for `r = 8, R = 3` it is 93. Beyond
`R = 3` the antisymmetry rarely yields lift on chess tokens because
near-puzzle tactics are typically 2- or 3-body conjunctions. The
`__init__` enforces `r in [1, 8]` and `max_grade in [1, r]`.

## Numerical stability

`z_i = tanh(W token_i)` is in `(-1, 1)`, so wedge products stay
bounded by `1`. The grade magnitudes are sqrt-clipped through
`clamp_min(eps)` to avoid NaN gradients at zero.

## Deferred internal proposals

The source packet (`external_36_exterior_product_rank1_resolvent_primitives.md`)
contains four other proposals; only the rank-1 (this primitive) is
promoted in this batch.

| Proposal | Status |
|---|---|
| `primitive_exterior_product_pool` (rank 1) | Implemented as p041. |
| `primitive_rank1_resolvent_pool` (rank 2) | Covered by p038 (Woodbury Set Resolver) in this same batch. |
| `primitive_orbit_stabilized_canonicalizer` (rank 3) | Variant of p036; the stabilizer-averaging extension is documented in p036's `ablations.md`. |
| `primitive_tropical_distance_transform` (rank 4) | Partial overlap with p039 (Occupancy Eikonal); deferred. |
| `primitive_capacitated_entropic_assignment` (rank 5) | Overlaps with existing Sinkhorn-style trunks; deferred. |

## Input contract

- Input: `simple_18` current-board tensor, shape `(B, 18, 8, 8)`.
- Output: `dict` with `logits` of shape `(B,)` plus the diagnostics
  listed in `architecture.md` and the i193 trunk diagnostics (`trunk_*`).
- The model rejects non-simple_18 inputs and non-1 `num_classes`,
  invalid `r`, `max_grade`, or `ablation`.

## Trainer compatibility

Same `(B, 18, 8, 8)` contract as i193. Returns `dict["logits"]` shape
`(B,)`. The trainer reads only `dict["logits"]` for loss.
