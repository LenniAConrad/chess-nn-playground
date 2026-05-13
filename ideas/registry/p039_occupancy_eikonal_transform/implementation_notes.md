# Implementation Notes

## Worktree

Implemented in the `cnp-primitive-gpt-orbit-boolean-algebra` worktree.

## Module

`src/chess_nn_playground/models/primitives/occupancy_eikonal_transform.py`

Contains:

- `OccupancyEikonalTransform` nn.Module.
- `build_occupancy_eikonal_transform_from_config`.
- `_king_neighbour_index()` -- static 64x8 neighbour table.
- `ALLOWED_ABLATIONS` tuple.

## Why softmin (logsumexp) instead of hard min

Hard min is non-differentiable and creates brittle gradients near
ties. Softmin (logsumexp with negated arguments) is smooth and matches
the temperature-controlled approximation used by `Vlastelica et al.`
for differentiable shortest path. The temperature `tau` (default
`0.5`) controls the bias / variance trade-off. Lower `tau` sharpens
the arrival-time field but increases gradient variance near argmin
ties.

## Why node-cost instead of edge-cost

Node-cost convention (`c_uv = c_v`, the destination-square cost) keeps
the parameter count tractable: one cost field per channel per square.
A full edge-cost convention would multiply by the 8 outgoing edges.
This matches standard fast-marching practice. If a future falsifier
flags directional anisotropy as needed, the deferred D2 extension in
`ablations.md` adds per-edge anisotropic costs.

## Numerical stability

`torch.logsumexp` is used throughout. The cost field is strictly
positive by construction (`softplus + cost_bias`). The relaxation is
bounded by `num_iterations` (default 6), which on an 8x8 grid is more
than enough for the king-neighbour diameter (7 hops worst case).

## Deferred internal proposals

The source packet (`external_34_active_esp_conflict_matching_eikonal_primitives.md`)
contains four other proposals which were *not* promoted in this batch:

| Proposal | Reason for deferral |
|---|---|
| `primitive_active_esp` (rank 1) | Duplicate of `p024_event_symmetric_interaction_accumulator`. |
| `primitive_conflict_matching_poly` (rank 2) | Truncated graph matching polynomial; combinatorial enumeration is expensive for general matroid constraints; deferred. |
| `primitive_clifford_accumulator` (rank 4) | Clifford geometric-product accumulator; requires precomputed multivector multiplication tables and grade projections; deferred (also has limited expected lift on 8x8 boards). |
| `primitive_stabilizer_orbitnorm` (rank 5) | Orbit-stabilised normalisation; partially overlaps with p036 (this batch's canonical-orbit primitive) and OrbitNorm-style ideas; deferred. |

## Input contract

- Input: `simple_18` current-board tensor, shape `(B, 18, 8, 8)`.
- Output: `dict` with `logits` of shape `(B,)` plus the diagnostics
  listed in `architecture.md` and the i193 trunk diagnostics (`trunk_*`).
- The model rejects non-simple_18 inputs and non-1 `num_classes`,
  invalid `q_channels`, `temperature`, `num_iterations`, or
  `ablation`.

## Trainer compatibility

Same `(B, 18, 8, 8)` contract as i193. Returns `dict["logits"]` shape
`(B,)`. The trainer reads only `dict["logits"]` for loss; diagnostics
are slice-report only.
