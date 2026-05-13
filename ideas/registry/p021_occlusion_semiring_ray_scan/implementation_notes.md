# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/occlusion_semiring_ray_scan.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Idea-local wrapper: `ideas/registry/p021_occlusion_semiring_ray_scan/model.py`.
- Registry key: `occlusion_semiring_ray_scan`.
- Source primitive: `ideas/research/primitives/external_16_ray_blocked_delta_pair_legal_edge_reduce.md`
  (rank-1 proposal `primitive_ray_blocked_scan`).

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Per-square tokens are built from the 12 piece planes plus the
side-to-move scalar.

## Transmittance: log-domain numerical stability

The mathematical definition is `T_{l} = prod_{q<l} (1 - O_{q})`. With
binary occupancy this product is exactly 0 or 1. The implementation
clamps the per-step `(1 - O)` to `[1e-4, 1]` before taking the log so
the cumulative sum is finite. The resulting `T` is then `exp(...) *
step_mask` so off-board cells contribute 0.

A naive `cumprod` implementation would lose precision after ~5-7
multiplications on long rays with non-binary `O`; the `log + cumsum`
path is numerically stable up to the floor.

## Ray geometry

Same shared `RayGeometry` lookup as p020 and p023:

```
ray_step_index: (8, 64, 7) long
ray_step_mask:  (8, 64, 7) float
```

## Stop-gradient contract

- `token_proj`, `direction_proj`, and the MLP heads receive gradients
  from the BCE loss.
- The occupancy tensor receives gradient through the
  `clamp(0, 1)` and `sum` over piece planes -- since the simple_18
  inputs are not learned, occupancy is effectively an integer-valued
  rule-derived feature.

## Output dict contract

- `logits` (rebound to `base_logit + primitive_delta`)
- `base_logit`
- `primitive_delta`, `primitive_delta_raw`
- `primitive_gate`, `primitive_gate_logit`, `primitive_gate_entropy`
- `osrs_mean_transmittance` -- mean of `T` across all `(d, s, l)`.
- `osrs_open_ray_fraction` -- fraction of ray cells with `T > 0.5`.

## Ablation modes

See `ablations.md` and `model.ALLOWED_ABLATIONS`. Primary falsifier:
`zero_occupancy`.

## Deferred internal proposals

Other proposals in the source packet that are *not* implemented here:

- `primitive_delta_pair_accumulator` -- sister to p022.
- `primitive_legal_edge_reduce` -- rule-generated edges.
- `primitive_orbit_action_norm` -- orbit normalisation.
- `primitive_soft_see_reducer` -- differentiable static-exchange.

## Why this is not a `ResearchPacketProbe` scaffold

Bespoke `nn.Module` wrapping the bespoke i193 trunk with explicit
log-domain transmittance computation and per-direction projection.
Does not delegate to a shared probe builder.
`implementation_kind: bespoke_model` is consistent with
`audit_implementation_kinds.py`.
