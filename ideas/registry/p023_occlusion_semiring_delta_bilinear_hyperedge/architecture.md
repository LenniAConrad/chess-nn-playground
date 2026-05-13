# Architecture

`Occlusion Semiring Delta-Bilinear Hyperedge` (p023) is an additive,
gated head over the existing i193
`ExchangeThenKingDualStreamNetwork` trunk. It combines a *backward*
ray semiring recurrence with a bilinear hyperedge contraction over
opposing-direction ray pairs.

## Mechanism

1. **i193 trunk forward**. Unchanged.

2. **Per-square token construction**. 12 piece planes + STM ->
   `x_s in R^{B, 64, token_dim}`.

3. **Ray gather**. Use shared `RayGeometry` lookup to gather per-step
   tokens and occupancy:
   - `ray_tokens (B, 8, 64, 7, token_dim)`
   - `ray_occ (B, 8, 64, 7)`

4. **Value projection + backward recurrence**.

   ```
   v_t = V * ray_token_t
   h <- (1 - O_t) * h + v_t,   walked from t = L-1 down to t = 0
   ```

   `h` ends as `h_{b, r, 0}` of shape `(B, 8, 64, hidden_dim)`. The
   `step_mask` zeroes out off-board contributions.

5. **Bilinear hyperedge contraction**. For each opposing-direction
   pair `(left, right)` in `[(N, S), (NE, SW), (E, W), (SE, NW)]`:

   ```
   left_emb  = W_L * h[:, left, :, :]
   right_emb = W_R * h[:, right, :, :]
   edge      = left_emb (.) right_emb            # Hadamard product
   ```

   The four `edge` tensors are concatenated along the feature axis
   to obtain `(B, 64, 4 * bilinear_dim)`.

6. **Pool + readout**. Mean-pool across the 64 squares to a
   `(B, 4 * bilinear_dim)` vector, feed through LayerNorm + GELU MLP
   to obtain `primitive_delta_raw`.

7. **Gate + fusion**. Sigmoid gate on the trunk joint feature, then
   `final_logit = base_logit + gate * primitive_delta_raw`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `zero_occupancy` | No blocker gate (`(1 - O) = 1`). Tests transmittance. |
| A2 | `uniform_occupancy` | Full blocker (`(1 - O) = 0`). Recurrence carries nothing. |
| A3 | `disable_bilinear` | Replace `left * right` with `left + right`. **Primary falsifier.** Tests the bilinear hyperedge claim. |
| A4 | `zero_delta` | Hold delta at 0. Recovers i193. |
| A5 | `trunk_only` | Strongest control. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine scores, and
any report-only metadata are not consumed.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass |
| Token + value projections | `O(64 * token_dim * 13 + 64 * 7 * hidden_dim * token_dim)` |
| Backward recurrence | `O(8 * 64 * 7 * hidden_dim)` (Python loop of 7 steps) |
| Bilinear hyperedge | `O(4 * 64 * hidden_dim * bilinear_dim)` |
| Readout MLP | LayerNorm + GELU MLP over `4 * bilinear_dim` |

## Implementation Binding

- Registered model name: `occlusion_semiring_delta_bilinear_hyperedge`.
- Source implementation: `src/chess_nn_playground/models/primitives/occlusion_semiring_delta_bilinear_hyperedge.py`.
- Shared ray geometry: `src/chess_nn_playground/models/primitives/ray_geometry.py`.
- Idea-local wrapper: `ideas/registry/p023_occlusion_semiring_delta_bilinear_hyperedge/model.py`.
- Training config: `ideas/registry/p023_occlusion_semiring_delta_bilinear_hyperedge/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.
