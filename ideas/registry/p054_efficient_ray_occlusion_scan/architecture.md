# Architecture

`Efficient Ray Occlusion Scan` (p054, EROS) is an additive, gated head
on top of the i193 `ExchangeThenKingDualStreamNetwork` trunk. The
operator runs a single tensorized scan over the legal queen-direction
ray representation (`RayGeometry`: 8 directions x 64 squares x 7 steps)
and preserves first / second blocker identity. There is no Python loop
over directions or steps in `forward`.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits `base_logit` and trunk diagnostics
   plus the joint pool feature (via `trunk_joint_features`).
2. **Per-square 16-channel feature**.
   - `us_occ`, `them_occ` -- sum of own / enemy piece planes (clamped).
   - `us_value`, `them_value` -- sum of fixed piece values (P=1, N=B=3,
     R=5, Q=9, K=200) weighted by the matching one-hot.
   - 6 us piece-type one-hots followed by 6 them piece-type one-hots.
3. **Occupancy mask**. `o_i = clamp(sum_p piece_plane_p(i), 0, 1)`.
   Ablation hooks may override `o` to all-zero (`empty_occupancy`),
   all-one (`uniform_occupancy`), or a batch-permuted version
   (`shuffle_occupancy`).
4. **Compact ray scan** (`ray_occlusion_scan`). Gather the 16-channel
   feature and occupancy along the 3584 padded ray slots, then compute
   the inclusive blocker prefix
   `k = cumsum(occ_ray, dim=-1)`. Equality tests on `k` and
   `k - occ_ray` produce four hard masks:

       visible    = 1[k - o == 0] * step_mask
       first      = o * 1[k == 1] * step_mask
       second     = o * 1[k == 2] * step_mask
       xray_lane  = 1[k - o == 1] * step_mask

   The first / second blocker feature summaries are masked reductions
   over the gathered feature tensor.
5. **Per-source-per-direction summaries** (13 channels per
   `(direction, source)` slot):

   - `visible_count`, `mobility_len`, `xray_lane_len`
   - `first_exists`, `first_value`, `first_us_occ`, `first_them_occ`
   - `second_exists`, `second_value`, `second_us_occ`, `second_them_occ`
   - `xray_pressure` (= `second_exists * second_value`)
   - `discovered_pressure + pinned_to_king`
     (=`first_us * second_them * second_value + first_them * them_king_second`)

6. **Direction-class projection**. Sum over directions weighted by
   `(ortho_mask, diag_mask, ortho_mask + diag_mask)` to obtain a per-
   square `(rook, bishop, queen) x summary` tensor. Mean + max pool
   over the 64 squares gives the readout vector.
7. **Delta head**. MLP on `cat(readout, joint)` to a scalar
   `primitive_delta_raw`.
8. **Gate**. MLP over `cat(joint, occ_density, mobility_mean,
   xray_mean)` to a sigmoid `primitive_gate`; initial bias
   `gate_init = -2.0`.
9. **Output**. `final_logit = base_logit + primitive_gate *
   primitive_delta_raw`.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full EROS architecture (default). |
| `first_only` | **Primary falsifier**. Zero everything beyond the first blocker (no second-blocker channels, no x-ray pressure, no discovered / pin candidates). |
| `no_blocker_id` | Zero only the side / value identity channels (first and second blocker). Keeps visibility / mobility / x-ray geometry but drops "what sits at the blocker". |
| `uniform_occupancy` | All squares treated as occupied. Only the first ray cell is ever visible. |
| `empty_occupancy` | Empty board: no blockers, just geometric ray length. |
| `shuffle_occupancy` | In-batch permutation of the occupancy mask. Decouples mask from position. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Trunk joint refeat | Two encoder passes total |
| Ray feature build | Per-square Linear-free composition of 16 channels |
| Compact scan | One `gather + cumsum` over ``(B, 8, 64, 7)`` and ``(B, 8, 64, 7, 16)`` |
| Direction-class pool | ``(B, S, 3 * 13)`` mean + max over 64 squares |
| Delta / gate | Small MLPs |

The scan body has no learnable parameters; only the delta and gate
heads do. At defaults (B=256) the per-step overhead is small compared
to the trunk and matches the cost profile of p020 / p021.

## Implementation Binding

- Registered model name: `efficient_ray_occlusion_scan`.
- Source implementation: `src/chess_nn_playground/models/primitives/efficient_ray_occlusion_scan.py`.
- Shared helpers: `RayGeometry` from
  `src/chess_nn_playground/models/primitives/ray_geometry.py`,
  `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p054_efficient_ray_occlusion_scan/model.py`.
- Training config: `ideas/registry/p054_efficient_ray_occlusion_scan/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `MODEL_SPECS["efficient_ray_occlusion_scan"] = (...)`.
