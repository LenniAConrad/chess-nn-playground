# Architecture

`Grassmann Rook-Matching Pool` (p043, GRMP) is an additive, gated head
on top of the i193 `ExchangeThenKingDualStreamNetwork` trunk. The
operator scores a bipartite attacker/defender token grid and returns
the truncated matching-polynomial coefficients with row/column
exclusion enforced algebraically.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits ``base_logit`` and trunk diagnostics
   plus the joint pool feature (recomputed via `trunk_joint_features`).
2. **Spatial feature**. Concatenate ex-stream and king-stream conv
   outputs: ``S in R^{B x 2 C_trunk x 8 x 8}``.
3. **Attacker / defender pools**. Two independent
   ``BoardTokenAttention`` modules produce
   ``a in R^{B x R x D}`` and ``d in R^{B x C x D}``.
4. **Validity masks**. Per-token sigmoid head -> ``m^a, m^d in [0, 1]``.
5. **Bilinear edge score**. ``z = tanh(Bilinear(a, d)) in R^{B x R x C x H}``.
6. **Matching coefficients**. ``grassmann_rook_matching_coefficients``
   returns ``e in R^{B x K x H}``. K = 1 closed form, K = 2 closed
   form, K = 3 via O(R*C) anchor-edge iteration.
7. **Coefficient norm**. LayerNorm over flat ``(B, K*H)``.
8. **Delta head**. MLP on `cat(e_flat, joint)` to a scalar
   `primitive_delta_raw`.
9. **Gate**. MLP over `cat(joint, atk_count, def_count, coeff_norm)`
   to sigmoid `primitive_gate`; initial bias ``gate_init = -2.0``.
10. **Output**. ``final_logit = base_logit + primitive_gate *
    primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full GRMP architecture (default). |
| `drop_exclusion` | **Primary falsifier**. Replace the rook-disjoint scan with an unrestricted elementary-symmetric pool over the flat edge tensor — same coefficients as p042 but no row/column constraint. |
| `scalar_score` | Collapse the H score channels to one (broadcast mean). Tests whether multi-channel edge representation matters. |
| `shuffle_attackers` | In-batch permutation of attacker tokens. Decouples attacker side from positions. |
| `shuffle_defenders` | In-batch permutation of defender tokens. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed by the model. The
operator depends only on the simple_18 board tensor (via the i193
trunk's spatial features).

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Trunk joint refeat | Two encoder passes total |
| Spatial features | One ex-stream + one king-stream forward (already shared with the trunk under default settings) |
| Attacker / defender pools | Two ``BoardTokenAttention`` instances |
| Bilinear edge score | One ``nn.Bilinear`` over (R*C*B) pairs |
| Coefficient scan | K=1, 2 closed form; K=3 O(R*C) anchor iteration |
| Delta / gate | Small MLPs |

At defaults (``R=C=8``, ``D=32``, ``H=8``, K=2, B=64), the edge tensor
is (B, 8, 8, 8) — under 1 MB. Head adds ~30k parameters at defaults.

## Implementation Binding

- Registered model name: `grassmann_rook_pool`.
- Source implementation: `src/chess_nn_playground/models/primitives/grassmann_rook_pool.py`.
- Shared helpers:
  - `BoardTokenAttention` from `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
  - `trunk_joint_features` from `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p043_grassmann_rook_pool/model.py`.
- Training config: `ideas/registry/p043_grassmann_rook_pool/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["grassmann_rook_pool"] = build_grassmann_rook_pool_from_config`.
