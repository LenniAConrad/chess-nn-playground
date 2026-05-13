# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/blocker_reset_ray_scan.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Idea-local wrapper: `ideas/registry/p020_blocker_reset_ray_scan/model.py`.
- Registry key: `blocker_reset_ray_scan`.
- Source primitive: `ideas/research/primitives/external_15_blocker_reset_edit_delta_fastweight.md`
  (rank-1 proposal `primitive_blocker_reset_scan`).

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Per-square tokens are computed by `_build_square_tokens` from
the 12 piece planes plus the side-to-move scalar. Occupancy is the
clamped sum of the 12 piece planes -- it is rule-derived and treated
as stop-gradient w.r.t. the input encoding.

## Ray geometry

The 8 queen directions and per-(direction, square, step) lookup are
provided by the shared `RayGeometry` module:

```
ray_step_index: (8, 64, 7) long  -- visited square per (d, s, l)
ray_step_mask:  (8, 64, 7) float -- 1 if (d, s, l) is on-board
```

Off-board slots are clamped to square index 0 and masked. This is the
same convention used by p021 and p023, so the three ray primitives
share the same geometry buffers if they coexist in a model.

## Stop-gradient contract

- Token projection (`token_proj`), input projection (`input_proj`),
  output projection (`output_proj`), and per-direction decay
  parameter (`decay_logit`) are all trainable.
- The blocker gate `(1 - O)` flows gradient through the occupancy
  computation `piece_planes.flatten(2).sum(dim=1).clamp(0, 1)`, but
  the simple_18 input is treated as fixed in the forward pass --
  there is no learnable parameter producing the occupancy.

## Output dict contract

The model output is a `dict[str, Tensor]` following the i193 contract,
extended with the standard primitive head keys (`base_logit`,
`primitive_delta`, `primitive_delta_raw`, `primitive_gate`,
`primitive_gate_logit`, `primitive_gate_entropy`) plus head-specific
diagnostics:

- `brrs_occupancy_density` -- mean occupancy per sample
- `brrs_ray_magnitude` -- RMS of the projected ray outputs
- `brrs_decay_mean` -- mean of sigmoid(lambda_d), broadcast across batch

## Ablation modes

See `ablations.md` and `model.ALLOWED_ABLATIONS`. The primary
falsifier is `zero_blocker`: forces the reset gate to 1 everywhere,
so the scan ignores blockers.

## Deferred internal proposals

The source packet contains four other proposals that are *not*
implemented here:

- `primitive_edit_delta_fastweight` -- sister to p019
  (reversible-delta kernel memory).
- `primitive_legal_edge_attention` -- legal-edge sparse attention.
- `primitive_rule_hyperedge_contract` -- rule-generated hyperedges.
- `primitive_chess_orbit_linear` -- chess-group equivariant linear.

## Why this is not a `ResearchPacketProbe` scaffold

Bespoke `nn.Module` wrapping the bespoke i193 trunk with explicit ray
scan recurrence and head MLPs. Does not delegate to a shared probe
builder. `implementation_kind: bespoke_model` is consistent with
`audit_implementation_kinds.py`.
