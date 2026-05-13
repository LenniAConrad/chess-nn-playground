# Architecture

`Event-Symmetric Interaction Accumulator` (p024) is an additive,
gated head over the existing i193
`ExchangeThenKingDualStreamNetwork` trunk. It supplies up to `R`-th
order Hadamard symmetric polynomial features over the active piece
set.

## Mechanism

1. **i193 trunk forward**. Unchanged.

2. **Per-square token construction**. 12 piece planes + STM ->
   `(B, 64, 13)` flat token input.

3. **Token projection**. Single learned linear layer maps each
   per-square 13-d feature to `token_dim`.

4. **Streaming recurrence**. Walk all 64 squares; at each square,
   update the per-order states:

   ```
   for r in [R, R-1, ..., 1]:
     if r == 1:
       E[0] += u
     else:
       E[r-1] += u * E[r-2]
   ```

   Empty squares contribute zero tokens (no-op).

5. **Normalisation**. With `normalize_by_active_count`, divide
   `E^{(r)}` by `|S|^r` so the readout sees scale-invariant features.

6. **Readout**. Feed `[E^{(1)}; ...; E^{(R)}]` through a LayerNorm +
   GELU MLP to obtain `primitive_delta_raw`.

7. **Gate + fusion**. Sigmoid gate on the trunk joint feature, then
   `final_logit = base_logit + gate * primitive_delta_raw`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `first_order_only` | Keep only `E^{(1)}` (set higher orders to 0). **Primary falsifier.** Collapses to EmbeddingBag-style sum. |
| A2 | `second_order_only` | Keep only `E^{(2)}` (zero out `E^{(1)}` and `E^{(3)}` if present). Tests the second-order term alone. |
| A3 | `shuffle_higher_orders` | In-batch permutation of `E^{(>=2)}`. Decouples higher orders from positions. |
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
| Streaming recurrence | `O(R * 64 * token_dim)` |
| Readout MLP | LayerNorm + GELU MLP over `R * token_dim` |

The streaming recurrence is `O(R * 64 * d)`. For `R = 2, d = 24`
this is ~3k FLOPs per sample, negligible compared to the trunk.

## Implementation Binding

- Registered model name: `event_symmetric_interaction_accumulator`.
- Source implementation: `src/chess_nn_playground/models/primitives/event_symmetric_interaction_accumulator.py`.
- Idea-local wrapper: `ideas/registry/p024_event_symmetric_interaction_accumulator/model.py`.
- Training config: `ideas/registry/p024_event_symmetric_interaction_accumulator/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.
