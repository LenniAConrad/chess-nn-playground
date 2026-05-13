# Architecture

`Differentiable Occupancy Eikonal Transform` (p039) is an additive,
gated head over the existing i193 trunk. It computes a soft arrival-
time field on the 8x8 king-neighbour graph and pools per-channel field
statistics into a logit delta.

## Mechanism

1. **i193 trunk forward**. The `ExchangeThenKingDualStreamNetwork`
   runs unchanged and emits `logits` (`base_logit`) and the dual-stream
   joint feature.

2. **Cost / seed projection**. Two linear maps project the joint
   feature to per-channel cost and seed fields, both shaped
   `(B, q_channels, 64)`. Each is passed through `softplus` and shifted
   by `cost_bias` so the cost field is strictly positive.

3. **King-neighbour gather buffer**. A static `(64, 8)` long tensor
   maps each square to its eight king-move neighbours, with
   out-of-board neighbours filled by self-loops.

4. **Soft Bellman-Ford relaxation**. Starting from `T = seed`, repeat
   `num_iterations` times:

   ```
   T_v = softmin_tau( s_v , { T_u + c_uv : (u, v) in E } )
   ```

   implemented as

   ```
   candidates[b, q, v, k] = T[neighbours[v, k]] + cost[b, q, v]
   alternatives           = concat(candidates, seed.unsqueeze(-1))
   T_new                  = -tau * logsumexp(-alternatives / tau, dim=-1)
   ```

5. **Pool**. Per-channel `field_mean`, `field_max`, `field_min` over
   the 64 squares.

6. **Readout MLP**. Concat the three pools (`R^{3 q_channels}`), run
   through LayerNorm + Linear + GELU + Dropout + Linear, output
   `primitive_delta_raw`.

7. **Gate**. Small MLP on the trunk joint feature with
   `gate_init = -2.0`. Effective gate is the sigmoid; the head starts
   as a no-op.

8. **Logit fusion**. `final_logit = base_logit + gate * delta_raw`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_field` | In-batch permutation of `T`. Primary falsifier. |
| A2 | `single_iteration` | Cap relaxation at 1 step. Tests whether propagation beyond the immediate king-neighbour ring matters. |
| A3 | `uniform_costs` | Force all `c_uv = cost_bias`. Tests cost-conditional contribution. |
| A4 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A5 | `trunk_only` | Alias of `zero_delta`. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass through the dual-stream encoder |
| Cost / seed projection | `O(feature_dim * q_channels * 64)` |
| Eikonal relaxation | `O(num_iterations * q_channels * 64 * 9)` per relaxation step |
| Readout | `O(q_channels + head_hidden_dim)` |

With `q_channels = 4`, `num_iterations = 6`: 6 * 4 * 64 * 9 = 13824 ops
for the relaxation -- well within trunk-overhead range.

## Implementation Binding

- Registered model name: `occupancy_eikonal_transform`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/occupancy_eikonal_transform.py`.
- Idea-local wrapper:
  `ideas/registry/p039_occupancy_eikonal_transform/model.py`.
- Training config:
  `ideas/registry/p039_occupancy_eikonal_transform/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.
