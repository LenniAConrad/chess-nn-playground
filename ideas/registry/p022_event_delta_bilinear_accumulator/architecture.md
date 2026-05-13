# Architecture

`Event-Delta Bilinear Accumulator` (p022) is an additive, gated head
over the existing i193 `ExchangeThenKingDualStreamNetwork` trunk. It
supplies a *second-order* sparse-set accumulator on top of i193's
first-order conv mixing, while keeping the pair-term cost at
`O(|S| d)` via the FM identity.

## Mechanism

1. **i193 trunk forward**. Unchanged. Emits `base_logit` and the
   joint pool feature.

2. **Per-square token construction**. 12 piece planes + STM scalar ->
   `(B, 64, 13)` flat token input.

3. **U / V projections**.

   ```
   U_s = W_U token_s   in R^{B, 64, bilinear_dim}
   V_s = W_V token_s   in R^{B, 64, bilinear_dim}
   ```

   Empty squares are masked to 0 via the occupancy mask.

4. **Accumulator sums**.

   ```
   A = sum_s U_s, B = sum_s V_s
   P = sum_s U_s (.) V_s
   Q = A (.) B - P                  # the pair term, FM identity
   ```

5. **Normalisation**. With `normalize_by_active_count`, divide `A, B`
   by the active piece count and `Q` by the active count squared so
   the head sees scale-invariant features.

6. **Readout**. Feed `[A; B; Q]` through a LayerNorm + GELU MLP to
   produce `primitive_delta_raw`.

7. **Gate + fusion**. A LayerNorm + GELU MLP on the trunk joint
   produces a sigmoid gate. `final_logit = base_logit + gate *
   primitive_delta_raw`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `first_order_only` | Drop the pair term `Q`. **Primary falsifier.** Collapses to a first-order accumulator. |
| A2 | `shuffle_pair_term` | In-batch permutation of `Q`. Decouples the pair signal from positions. |
| A3 | `zero_delta` | Hold delta at 0. Recovers i193. |
| A4 | `trunk_only` | Strongest control. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine scores, and
any report-only metadata are not consumed.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass |
| U / V projections | `O(64 * bilinear_dim * 13)` |
| Sums and FM identity | `O(64 * bilinear_dim)` |
| MLP heads | LayerNorm + GELU MLPs over `3 * bilinear_dim` |

The dominant head cost is the U / V projections; the FM identity
turns the would-be `O(64^2 d)` pair sum into `O(64 d)`.

## Implementation Binding

- Registered model name: `event_delta_bilinear_accumulator`.
- Source implementation: `src/chess_nn_playground/models/primitives/event_delta_bilinear_accumulator.py`.
- Idea-local wrapper: `ideas/registry/p022_event_delta_bilinear_accumulator/model.py`.
- Training config: `ideas/registry/p022_event_delta_bilinear_accumulator/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.
