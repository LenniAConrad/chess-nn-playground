# Architecture

`Occlusion-Aware Ray Scan Head` (p029) is an additive, gated head over
the i193 trunk. Promoted from the first-ranked proposal of
`ideas/research/primitives/external_26_delta_update_occlusion_ray_piece_kernels.md`.

The model consumes the `simple_18` `(B, 18, 8, 8)` board tensor and
returns one puzzle logit plus the standard primitive diagnostics dict.

## Mechanism

1. **i193 trunk forward**. Emits the canonical diagnostics.
2. **Per-square features**. `feature_proj` (1x1 conv) projects the 12
   piece planes to a per-square feature stack of width `feature_dim`.
3. **Blocker gate**. `blocker_gate` (linear `(feature_dim ->
   NUM_DIRECTIONS)`) emits per-(square, direction) gate logits;
   sigmoid yields the in-range blocker coefficient `g_{i, d}`.
4. **Selective scan**. For each direction:
   - Initialise `state = zeros`.
   - For `_` in range(max_ray_length): shift state by `(dr, df)`,
     multiply by `g`, add the per-square features.
   - Project the per-direction state with `direction_proj` and
     mean-pool over the board.
5. **Fuse**. Stack the per-direction `(B, feature_dim)` outputs,
   LayerNorm, and concatenate with the four trunk diagnostics.
6. **Additive logit**. Two MLPs over the fusion vector emit
   `primitive_delta_raw` and `gate_logit`; `final_logit = base_logit +
   sigmoid(gate_logit) * primitive_delta_raw`.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and any other report-only metadata are *not* consumed. The blocker
gate is a function of the per-square features projected from the
simple_18 piece planes only.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| `feature_proj` | `B * 12 * feature_dim * 64` MACs |
| `blocker_gate` | `B * feature_dim * NUM_DIRECTIONS * 64` MACs |
| Selective scan | `B * NUM_DIRECTIONS * max_ray_length * feature_dim * 64` MACs |
| Per-direction projection | `B * NUM_DIRECTIONS * feature_dim^2` MACs |
| Fusion MLP | Two MLPs over `(NUM_DIRECTIONS * feature_dim + 4)` |

At default `feature_dim=16` the scan is bounded by ~7300 FLOPs per
direction per sample.

## Deferred external_26 proposals

- **DUA** (Delta-Update Accumulator) — overlaps with `p025` / `p028`.
- **EPIK** (Equivariant Piece-Identity Kernels) — trunk-level
  weight-tying primitive.
- **LMMP** (Legal-Move Manifold Projection) — overlaps with `p027`.
- **DBI** (Differentiable Bitwise Interaction) — separate bit-logic
  primitive outside this batch.

## Implementation Binding

- Registered model name: `occlusion_aware_ray_scan_head`.
- Source implementation: `src/chess_nn_playground/models/primitives/occlusion_aware_ray_scan_head.py`.
- Idea-local wrapper: `ideas/registry/p029_occlusion_aware_ray_scan_head/model.py`.
- Training config: `ideas/registry/p029_occlusion_aware_ray_scan_head/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["occlusion_aware_ray_scan_head"] = build_occlusion_aware_ray_scan_head_from_config`.
