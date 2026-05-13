# Architecture

`Occlusion Semiring Ray Scan` (p021) is an additive, gated head over
the existing i193 `ExchangeThenKingDualStreamNetwork` trunk. The
thesis (see `math_thesis.md`) is that the i193 trunk's conv mixing
layers see through occupied squares unrealistically -- a 3x3 conv at
a depth-2 trunk does not natively respect "rook ray stops at first
blocker". p021 supplies an exclusive prefix-product transmittance
operator whose per-cell visibility weight matches the chess sliding
rule.

## Mechanism

1. **i193 trunk forward**. Unchanged. Emits `base_logit` and the
   joint pool feature.

2. **Per-square token construction**. 12 piece planes + STM plane ->
   `x_s in R^{B, 64, token_dim}` via a single linear layer.

3. **Transmittance computation**. Use
   `_compute_transmittance(occupancy, ray_step_index, ray_step_mask)`:

   ```
   log_term = log(clamp(1 - O_along_ray, eps, 1))
   exclusive = shift_right_one_step(cumsum(log_term))
   T = exp(exclusive) * step_mask
   ```

   Shapes: `T in (B, 8, 64, 7)`. Off-board steps are 0.

4. **Ray-token gather + per-direction projection**.
   - `ray_tokens = gather_along_rays(x, ray_step_index, ray_step_mask)`
     of shape `(B, 8, 64, 7, token_dim)`.
   - The `direction_proj` layer stores `(NUM_DIRECTIONS * hidden_dim,
     token_dim)` weights and `(NUM_DIRECTIONS * hidden_dim)` biases.
     We reshape to `(8, hidden_dim, token_dim)` and apply via einsum
     so each direction `d` gets its own projection `A_d`.

5. **Transmittance-weighted reduction**.

   ```
   y_{b, d, s} = sum_{l=1..L} T_{b, s, d, l} * (A_d x)_{l}
   ```

6. **Square pooling + readout**. Concatenate directions and mean-pool
   across the 64 squares to a `(B, 8 * hidden_dim)` vector, feed
   through LayerNorm + GELU MLP to obtain `primitive_delta_raw`.

7. **Gate + fusion**. A LayerNorm + GELU MLP on the trunk joint
   produces a sigmoid gate. `final_logit = base_logit + gate *
   primitive_delta_raw`. Gate is initialised near closed.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `zero_occupancy` | Treat the board as empty. The transmittance becomes 1 everywhere so ray cells contribute equally. **Primary falsifier.** |
| A2 | `uniform_occupancy` | Treat every square as occupied. Only step 1 has `T = 1`; deeper steps zero out. Tests that depth carries signal. |
| A3 | `isotropic_A` | Share the per-direction projection across all 8 directions. Tests whether direction-specific parameters help. |
| A4 | `zero_delta` | Hold delta at 0. Recovers i193. |
| A5 | `trunk_only` | Strongest control. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine scores, and
any report-only metadata are not consumed.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass |
| Token projection | `O(64 * token_dim * 13)` |
| Transmittance (log-domain cumsum) | `O(8 * 64 * 7)` |
| Per-direction projection | `O(8 * 64 * 7 * token_dim * hidden_dim)` |
| Transmittance-weighted reduce | `O(8 * 64 * 7 * hidden_dim)` |
| Readout MLP | LayerNorm + GELU MLP over `8 * hidden_dim` |

The dominant cost is the per-direction projection einsum. Throughput
should be within ~15% of i193 at scout scale.

## Implementation Binding

- Registered model name: `occlusion_semiring_ray_scan`.
- Source implementation: `src/chess_nn_playground/models/primitives/occlusion_semiring_ray_scan.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Idea-local wrapper: `ideas/registry/p021_occlusion_semiring_ray_scan/model.py`.
- Training config: `ideas/registry/p021_occlusion_semiring_ray_scan/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.
