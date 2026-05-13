# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/occlusion_semiring_delta_bilinear_hyperedge.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Idea-local wrapper: `ideas/registry/p023_occlusion_semiring_delta_bilinear_hyperedge/model.py`.
- Registry key: `occlusion_semiring_delta_bilinear_hyperedge`.
- Source primitive: `ideas/research/primitives/external_19_occlusion_semiring_delta_bilinear_hyperedge.md`
  (rank-1 proposal `primitive_1_occlusion_semiring_scan`, with the
  file-name promise of a "delta_bilinear_hyperedge" step layered on
  top).

## Inputs

The model only consumes the `simple_18` `(B, 18, 8, 8)` current-board
tensor. Per-square tokens are built from the 12 piece planes plus
the side-to-move scalar.

## Backward recurrence

The recurrence walks the ray from depth `L - 1` down to `0`:

```python
h = zeros(B, 8, 64, hidden_dim)
for t in range(RAY_MAX_LEN - 1, -1, -1):
    gate = (1.0 - ray_occ[..., t]) * ray_step_mask[..., t]
    value = V * ray_tokens[..., t, :] * mask[..., t]
    h = gate.unsqueeze(-1) * h + value
```

After the loop, `h` represents `h_{b, r, 0}`. This differs from
p020's *forward* recurrence and p021's non-recurrent *forward*
exclusive prefix product.

## Bilinear hyperedge pairs

The four opposing-direction pairs:

| index | left | right |
|---|---|---|
| 0 | N (0) | S (4) |
| 1 | NE (1) | SW (5) |
| 2 | E (2) | W (6) |
| 3 | SE (3) | NW (7) |

The hyperedge embedding for pair `p` at square `s` is

```
edge_{b, s, p} = W_L * h_{b, left_p, s} (.) W_R * h_{b, right_p, s}
```

`W_L` and `W_R` are shared across pairs (one of each in the head),
which matches the chess geometric invariant: there is no a-priori
reason to prefer one diagonal over another for the bilinear
contraction.

## Stop-gradient contract

Standard: all projections and head MLP weights are trainable;
occupancy is a rule-derived clamp of the input encoding.

## Output dict contract

- `logits`, `base_logit`, `primitive_delta`, `primitive_delta_raw`
- `primitive_gate`, `primitive_gate_logit`, `primitive_gate_entropy`
- `osdb_hidden_magnitude` -- mean ||h||^2 over (B, 8, 64, hidden_dim).
- `osdb_pair_hyperedge_magnitude` -- mean over the 4 pair magnitudes.

## Ablation modes

See `ablations.md` and `model.ALLOWED_ABLATIONS`. Primary falsifier:
`disable_bilinear`.

## Deferred internal proposals

Other proposals in the source packet that are *not* implemented here:

- `primitive_2_delta_bilinear_accumulator` -- covered by p022.
- `primitive_3_legal_hyperedge_contraction`.
- `primitive_4_tropical_threat_scan`.
- `primitive_5_chess_orbit_linear`.

## Why this is not a `ResearchPacketProbe` scaffold

Bespoke `nn.Module` wrapping the bespoke i193 trunk with an explicit
Python-loop backward recurrence and a bilinear hyperedge contraction.
Does not delegate to a shared probe builder.
`implementation_kind: bespoke_model` is consistent with
`audit_implementation_kinds.py`.
