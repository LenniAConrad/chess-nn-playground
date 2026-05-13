# Architecture

`Ray-Parallel SSM Head` (p030) is an additive, gated head over the
i193 trunk. Promoted from the first-ranked proposal of
`ideas/research/primitives/external_27_ray_parallel_ssm_delta_accumulator_sparse_conv.md`.

The model consumes the `simple_18` `(B, 18, 8, 8)` board tensor and
returns one puzzle logit plus the standard primitive diagnostics dict.

## Mechanism

1. **i193 trunk forward**. Emits the canonical diagnostics.
2. **Per-square features**. `feature_proj` (1x1 conv) projects the 12
   piece planes to a per-square feature stack of width `feature_dim`.
3. **Per-(square, direction, channel) A and B**. `A_proj` and `B_proj`
   are linear maps that emit `NUM_DIRECTIONS * feature_dim` scalars
   per square. After reshape they are `(B, NUM_DIRECTIONS, F, 8, 8)`
   tensors in (0, 1) via sigmoid.
4. **Per-direction selective scan**. For each direction:
   - Initialise `state = zeros`.
   - For `_` in range(max_ray_length): shift state along the
     direction, multiply by `A[..., d]`, add `B[..., d] * features`.
   - Project the per-direction state by `C[d]` (a learned per-channel
     scalar) and accumulate into `y_total`.
5. **Pool + fuse**. Mean-pool `y_total` to `(B, feature_dim)`,
   LayerNorm, concat with the four trunk diagnostics, run gate /
   delta MLPs.
6. **Additive logit**. `final_logit = base_logit +
   sigmoid(gate_logit) * primitive_delta_raw`.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and any other report-only metadata are *not* consumed. A, B, and C
are derived from the simple_18 piece planes (via the feature stack)
plus the learned C parameter.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| `feature_proj` | `B * 12 * feature_dim * 64` MACs |
| `A_proj`/`B_proj` | `B * feature_dim * NUM_DIRECTIONS * feature_dim * 64` MACs each |
| Scan | `B * NUM_DIRECTIONS * max_ray_length * feature_dim * 64` MACs |
| Fusion MLP | Two MLPs over `(feature_dim + 4)` |

At default `feature_dim=16`, the A/B projections are the dominant
per-step cost in the head.

## Deferred external_27 proposals

- **DDA** (Differentiable Delta-Accumulator) — overlaps with `p025`
  / `p028`.
- **move_gated_conv** — overlaps with `p027`.
- **involution_sym** — trunk-level weight-tying primitive.
- **soft_logic_gate** (DBLA) — separate bit-logic primitive outside
  this batch.

## Implementation Binding

- Registered model name: `ray_parallel_ssm_head`.
- Source implementation: `src/chess_nn_playground/models/primitives/ray_parallel_ssm_head.py`.
- Idea-local wrapper: `ideas/registry/p030_ray_parallel_ssm_head/model.py`.
- Training config: `ideas/registry/p030_ray_parallel_ssm_head/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["ray_parallel_ssm_head"] = build_ray_parallel_ssm_head_from_config`.
