## Architecture

`Pin / X-ray / Skewer` (p049, PXS) is an additive, gated head on top
of the i193 `ExchangeThenKingDualStreamNetwork` trunk. The thesis
(see `math_thesis.md`) is that the i193 trunk's 3x3 convolutions and
the existing ray heads (p020, p021, p034) leave the *ordered* slider
geometry under-exploited: which square is the *first* enemy, which
square is the *second*, and what their piece-types are. p049 makes
those facts a typed event tensor that the trunk gates and adds.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits `base_logit` and trunk diagnostics
   plus the joint pool feature (recomputed via `trunk_joint_features`
   without firing the trunk's heads twice).
2. **Mover-oriented piece state**. The simple_18 piece planes are
   re-oriented by the side-to-move plane 12 so channels 0-5 hold our
   pieces and 6-11 hold the opponent's. Occupancy is the clamped sum
   of all 12 absolute piece planes.
3. **Per-direction slider activation**. Queens fire in all 8
   directions; rooks fire in the 4 orthogonal directions; bishops
   fire in the 4 diagonal directions. The result `src` is shape
   `(B, 8, 64)` per source square per direction.
4. **Ordered blocker masks**. Using the shared `ray_geometry`
   `(8, 64, 7)` index/mask tables, gather occupancy and per-piece
   indicators along each ray. A `cumsum` over the 7-step axis
   produces `first_occ` (cumulative-occupancy == 1) and `second_occ`
   (cumulative-occupancy == 2) masks. These are then intersected
   with `us_any` / `them_any` / `them_king` / `them_queen` /
   `them_rook` / `them_value` / `us_value` to get the typed first /
   second occupant masses per ray step.
5. **Event masses** (per source square, per direction). The six
   typed events are:

   | event             | formula (per ray, abbreviated)                       |
   |-------------------|------------------------------------------------------|
   | `xray1`           | `src * (first_them + first_us) * second_value`       |
   | `abs_pin`         | `src * first_them * second_king`                     |
   | `rel_pin`         | `src * first_them * (second_queen + 0.6*second_rook)`|
   | `discovered`      | `src * first_us * second_value`                      |
   | `skewer`          | `src * second_any * relu(first_value - second_value)`|
   | `pinned_defender` | `src * first_value * second_king`                    |

   `*_per_ray` quantities are step-summed before multiplication. All
   tensors are non-negative, so the per-ray sum is well-defined.
6. **Per-square channels**. Direction-summed to `(B, 6, 64)`,
   element-wise scaled by `sigmoid(event_scale)` (one learnable
   scalar per event channel).
7. **12-dim summary**. `cat(mean_over_squares, max_over_squares)`.
8. **Delta head**. `LayerNorm + Linear + GELU + Dropout + Linear`
   on `cat(joint, summary)` to a scalar `primitive_delta_raw`.
9. **Gate**. MLP over `cat(joint, event_total_mean)` to a sigmoid
   `primitive_gate`; initial bias `gate_init = -2.0` so the primitive
   starts near-closed.
10. **Output**. `final_logit = base_logit + primitive_gate *
    primitive_delta_raw`.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full PXS architecture (default). |
| `no_xray1` | **Primary falsifier**. Zero every event term that depends on `second_occ` (the one-blocker x-ray channel). If `pin` / `skewer` / `discovered_attack` slice lift survives, the operator was not actually using x-ray-through-one-blocker logic. |
| `uniform_values` | Replace the learnable piece-value softmax with a uniform `1/6` field. Tests whether the king / queen / rook value context is load-bearing. |
| `no_pin_def` | Zero the pinned-defender event channel. Tests whether the additional defender-load proxy buys anything. |
| `shuffle_rays` | In-batch permutation of the `(8, 64, 7)` ray-index table. Decouples the rule-derived ray geometry from the position. |
| `zero_delta` | Zero the primitive delta. Recovers the i193 baseline numerically. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are **not** consumed by the model. The operator depends only on the
simple_18 piece-presence planes plus the side-to-move plane.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder. |
| Trunk joint refeat | One additional encoder pass for the joint feature. |
| Per-direction slider activation | `O(8 * 64)` per sample. |
| Ray gather | `O(7 * 8 * 64)` per scalar channel; 8 scalar channels. |
| Cumsum + masks | `O(8 * 64 * 7)` per sample. |
| Direction-sum + 6-channel scale | `O(6 * 64)`. |
| Delta head + gate | Small MLPs. |

There are **no training-time Python loops** over rays, sources, or
step lengths. The whole event builder is gather + cumsum + boolean
masks + sum. Geometry buffers carry zero parameters; the head adds
~12k parameters at defaults.

## Implementation Binding

- Registered model name: `pin_xray_skewer`.
- Source implementation: `src/chess_nn_playground/models/primitives/pin_xray_skewer.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Shared trunk helper: `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p049_pin_xray_skewer/model.py`.
- Training config: `ideas/registry/p049_pin_xray_skewer/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `'pin_xray_skewer': ('chess_nn_playground.models.primitives.pin_xray_skewer', 'build_pin_xray_skewer_from_config')`.
- Source research markdown: `ideas/research/primitives/external_44_pin_xray_skewer_primitive.md`.
