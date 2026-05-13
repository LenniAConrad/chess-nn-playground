# Architecture

`Kirchhoff Mobility Solve` (p045, KMS) is an additive, gated head on
top of the i193 `ExchangeThenKingDualStreamNetwork` trunk. The
operator solves the SPD equilibrium of a learned grid-graph
conductance system and pools the resulting per-square potentials.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits ``base_logit`` and trunk diagnostics
   plus the joint pool feature (via `trunk_joint_features`).
2. **Spatial feature**. Concatenate ex-stream and king-stream conv
   outputs: ``S in R^{B x 2C x 8 x 8}``.
3. **Per-square embedding**. Flatten ``S`` to ``X in R^{B x 64 x 2C}``.
4. **Source head**. ``Linear(2C, source_channels) -> s in R^{B x 64 x source_channels}``.
5. **Edge endpoint feature**. Gather ``X`` at head/tail of each of the
   112 grid edges, concatenate to ``(B, 112, 4C)``.
6. **Conductance head**. Two-layer MLP -> ``c = softplus(.) in R_+^{B x 112}``.
7. **Laplacian assembly**. ``L_b = D^T diag(c_b) D + shift * I``.
8. **SPD solve**. ``u_b = solve(L_b, s_b) in R^{64 x source_channels}``.
9. **Output projection**. ``Y_v = u_v W_o in R^{output_channels}``.
10. **Pool**. Mean + max per channel -> ``(B, 2 * output_channels)``.
11. **Delta head**. MLP on `cat(comp_feat, joint)` to a scalar
    `primitive_delta_raw`.
12. **Gate**. MLP over `cat(joint, kms_potential_norm,
    kms_conductance_mean)` to sigmoid `primitive_gate`; initial bias
    ``gate_init = -2.0``.
13. **Output**. ``final_logit = base_logit + primitive_gate *
    primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full KMS architecture (default). |
| `uniform_conductance` | **Primary falsifier**. Replace the learned conductance with all-ones. The resolvent becomes a fixed linear map of the source. |
| `diagonal_only` | Drop the Laplacian term entirely; ``u = s / shift``. Tests whether the resolvent structure matters beyond a per-square source readout. |
| `shuffle_conductance` | In-batch permutation of the conductance vector. Decouples conductance from board geometry. |
| `zero_source` | Zero the source term; ``u = 0``. Sanity check. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed by the model. The 8x8
grid graph is fixed and registered as a non-trainable buffer.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Trunk joint refeat | Two encoder passes total |
| Spatial features | One ex + one king encoder forward |
| Source / conductance heads | Linear / 2-layer MLP |
| Laplacian assembly | O(B * V * E) einsum |
| SPD solve | O(B * 64^3) |
| Output / delta / gate | Linear + MLPs |

At defaults (``source_channels=6``, ``output_channels=8``, B=64), the
SPD solve runs in well under 1 ms on RTX 3070-class hardware. Head
adds ~15k parameters at defaults.

## Implementation Binding

- Registered model name: `kirchhoff_mobility_solve`.
- Source implementation: `src/chess_nn_playground/models/primitives/kirchhoff_mobility_solve.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p045_kirchhoff_mobility_solve/model.py`.
- Training config: `ideas/registry/p045_kirchhoff_mobility_solve/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["kirchhoff_mobility_solve"] = build_kirchhoff_mobility_solve_from_config`.
