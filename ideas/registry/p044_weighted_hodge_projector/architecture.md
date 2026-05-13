# Architecture

`Weighted Hodge Projector` (p044, WHP) is an additive, gated head on
top of the i193 `ExchangeThenKingDualStreamNetwork` trunk. The
operator decomposes a learned edge flow over the 8x8 board grid into
gradient + curl + harmonic components via two batched SPD solves.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits ``base_logit`` and trunk diagnostics
   plus the joint pool feature (via `trunk_joint_features`).
2. **Spatial feature**. Concatenate ex-stream and king-stream conv
   outputs: ``S in R^{B x 2C x 8 x 8}``.
3. **Edge endpoint features**. For each of the 112 oriented edges, gather
   ``S`` at the head and tail squares and concatenate.
4. **Edge projection**. ``edge_proj`` (Linear + GELU) -> ``(B, 112,
   edge_feature_dim)``.
5. **Flow & metric heads**.
   - ``flow_head`` (Linear) -> ``F in R^{B x 112 x flow_channels}``.
   - ``metric_head`` (Linear) -> ``w = softplus(.) in R_+^{B x 112}``.
6. **Hodge decomposition**. ``hodge_decompose(F, w, D_0, D_1, eps)``
   returns ``(G, Cr, H)``.
7. **Per-component energy summary**. Mean+max per channel across edges
   for each of the three components -> ``(B, 3 * 2 * flow_channels)``.
8. **Delta head**. MLP on `cat(comp_feat, joint)` to scalar
   `primitive_delta_raw`.
9. **Gate**. MLP over `cat(joint, g_energy, c_energy, h_energy)` to
   sigmoid `primitive_gate`; initial bias `gate_init = -2.0`.
10. **Output**. ``final_logit = base_logit + primitive_gate *
    primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full WHP architecture (default). |
| `uniform_metric` | **Primary falsifier**. Set ``W = I``; the projection becomes a fixed linear map. |
| `drop_curl` | Zero the curl component. Tests whether circulation matters. |
| `drop_gradient` | Zero the gradient component. |
| `drop_harmonic` | Zero the harmonic residual. |
| `shuffle_edge_flow` | In-batch permutation of the per-edge flow tensor. Decouples flow values from edge geometry. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed. The 8x8 grid complex is
fixed (registered as a non-trainable buffer).

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Trunk joint refeat | Two encoder passes total |
| Spatial features | One ex + one king encoder forward |
| Edge endpoint gather | O(B * 112 * 2C) |
| Flow / metric heads | Linear over `(B, 112, d_edge)` |
| Vertex Laplacian solve | O(B * 64^3) SPD solve |
| Face Laplacian solve | O(B * 49^3) SPD solve |
| Delta / gate | Small MLPs |

At defaults (``flow_channels=4``, ``edge_feature_dim=16``, B=64), the
solves are well under 1 ms each on RTX 3070-class hardware. Head adds
~20k parameters at defaults.

## Implementation Binding

- Registered model name: `weighted_hodge_projector`.
- Source implementation: `src/chess_nn_playground/models/primitives/weighted_hodge_projector.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p044_weighted_hodge_projector/model.py`.
- Training config: `ideas/registry/p044_weighted_hodge_projector/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["weighted_hodge_projector"] = build_weighted_hodge_projector_from_config`.
