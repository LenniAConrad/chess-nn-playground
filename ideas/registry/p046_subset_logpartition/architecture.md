# Architecture

`Bounded Subset Log-Partition Transform` (p046, SLPT) is an additive,
gated head on top of the i193 `ExchangeThenKingDualStreamNetwork`
trunk. The operator runs the log-semiring elementary-symmetric scan
over the 64 squares and pools the K log-partition values per channel.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits ``base_logit`` and trunk diagnostics
   plus the joint pool feature (via `trunk_joint_features`).
2. **Per-square log-weight**. Each square gets a (12 piece-presence +
   1 side-to-move) descriptor; ``log_weight_proj`` projects to
   ``log_weight_dim`` and LayerNorms (no tanh; values are real and
   live in log domain).
3. **Occupancy mask**. ``m_i = 1[sum_p piece_plane_p(i) > 0]``.
4. **Log-semiring scan**. ``subset_logpartition_scan`` returns
   ``Y in R^{B x K x C}``. Default ``K = 3``.
5. **Head input clip**. Clamp to ``[-30, 30]`` to bound the head input
   across positions of very different active-token counts.
6. **Coefficient norm**. LayerNorm over flat (B, K*C).
7. **Delta head**. MLP on `cat(Y_flat, joint)` to a scalar
   `primitive_delta_raw`.
8. **Gate**. MLP over `cat(joint, active_mean, logpartition_norm)`
   to sigmoid `primitive_gate`; initial bias ``gate_init = -2.0``.
9. **Output**. ``final_logit = base_logit + primitive_gate *
   primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full SLPT architecture (default, K=3). |
| `k1_only` | **Primary falsifier**. Effective K := 1; collapses to logsumexp pool. |
| `uniform_mask` | ``m_i := 1`` for all squares. Tests whether the chess-rule occupancy mask matters. |
| `shuffle_mask` | In-batch permutation of the occupancy mask. Decouples mask from position. |
| `shuffle_tokens` | Permute square order. Coefficient values are exactly invariant. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Trunk joint refeat | Two encoder passes total |
| Log-weight projector | Linear + LayerNorm |
| Log-semiring scan | O(64 * K * log_weight_dim) ``logaddexp`` ops |
| Delta / gate | Small MLPs |

At defaults (``log_weight_dim=32``, ``K=3``, B=256) the per-step
overhead is small compared to the trunk. Head adds ~80k parameters.

## Implementation Binding

- Registered model name: `subset_logpartition`.
- Source implementation: `src/chess_nn_playground/models/primitives/subset_logpartition.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p046_subset_logpartition/model.py`.
- Training config: `ideas/registry/p046_subset_logpartition/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["subset_logpartition"] = build_subset_logpartition_from_config`.
