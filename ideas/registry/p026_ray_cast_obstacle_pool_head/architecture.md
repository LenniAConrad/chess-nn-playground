# Architecture

`Ray-Cast Obstacle Pooling Head` (p026) is an additive, gated head over
the i193 trunk. Promoted from the first-ranked proposal of
`ideas/research/primitives/external_22_ray_cast_obstacle_pooling_sparse_emit.md`.

The model consumes the `simple_18` `(B, 18, 8, 8)` current-board tensor
and returns one puzzle logit plus the standard primitive diagnostics
dict.

## Mechanism

1. **i193 trunk forward**. Emits the canonical diagnostics.
2. **Per-square features**. A 1x1 conv reduces the 12 piece planes to a
   per-square feature stack of width `feature_dim`.
3. **Rule-exact occupancy**. `occupancy_from_simple_18(board)` sums the
   12 piece planes and clamps to `[0, 1]` — this is the blocker mask
   used inside RayPool.
4. **8-direction prefix scan**. For each of the 8 cardinal/diagonal
   directions, the head shifts the per-square features by `(s * dr, s *
   df)` for `s = 1..max_ray_length`, multiplies by the running unblocked
   coefficient and the learned per-direction decay, and accumulates the
   result. The blocker mask is updated at each step using the occupancy
   at the just-visited target square.
5. **Pool + fuse**. The per-direction `(feature_dim, 8, 8)` tensors are
   mean-pooled over the board, flattened across directions, LayerNorm'd,
   and concatenated with the 4 trunk diagnostics.
6. **Additive logit**. Two MLPs over the fusion vector emit
   `primitive_delta_raw` and `gate_logit`. The final logit is `base_logit
   + sigmoid(gate_logit) * primitive_delta_raw`.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and any other report-only metadata are *not* consumed. The blocker mask
and the per-square features are read from `simple_18` only.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| RayPool scan | `O(8 * max_ray_length * feature_dim * 64)` |
| Fusion head | Two small MLPs over a `(8 * feature_dim + 4)`-d vector |

At the defaults (`feature_dim=16`, `max_ray_length=7`) the scan is
~7300 FLOPs per sample per direction — small compared to the i193 trunk
forward.

## Deferred external_22 proposals

The research file contains five proposals. We implement RayPool only:

- **DeltaGELU** — a stateful non-linearity cache that does not fit our
  stateless `model(x)` contract.
- **LegalMoveAttn** — overlaps with the SLMR primitive (`p027`).
- **ZeroSumExchange** — a routing constraint that would replace the
  trunk rather than supplement it.
- **SparseEmitLinear** — a kernel-level optimisation rather than a
  representation primitive.

## Implementation Binding

- Registered model name: `ray_cast_obstacle_pool_head`.
- Source implementation: `src/chess_nn_playground/models/primitives/ray_cast_obstacle_pool_head.py`.
- Idea-local wrapper: `ideas/registry/p026_ray_cast_obstacle_pool_head/model.py`.
- Training config: `ideas/registry/p026_ray_cast_obstacle_pool_head/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["ray_cast_obstacle_pool_head"] = build_ray_cast_obstacle_pool_head_from_config`.
