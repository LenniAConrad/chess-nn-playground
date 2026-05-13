# Architecture

`Truncated Multiset Polynomial Pool` (p042, TMPP) is an additive,
gated head on top of the i193 `ExchangeThenKingDualStreamNetwork`
trunk. The operator scans elementary-symmetric polynomial coefficients
``e_1, ..., e_K`` of a per-square latent over the occupancy mask,
producing a low-order coalition spectrum that is concatenated with
the trunk joint pool feature before delta projection.

The model consumes `simple_18` `(B, 18, 8, 8)` and returns one puzzle
logit plus a per-sample diagnostics dict.

## Forward pass

1. **i193 trunk forward**. Emits ``base_logit`` and trunk diagnostics
   plus the joint pool feature (recomputed via `trunk_joint_features`
   without firing the trunk heads twice).
2. **Per-square descriptor**. Each square gets a (12 piece-presence +
   1 side-to-move) descriptor; ``token_proj`` projects to ``latent_dim``
   followed by LayerNorm and tanh.
3. **Occupancy mask**. ``m_i = 1[sum_p piece_plane_p(i) > 0]``.
4. **Polynomial scan**. ``truncated_elementary_symmetric_scan`` returns
   ``e in R^{B x K x C}``. The default ``K = 3`` matches the
   source primitive's recommended cap.
5. **Coefficient normalisation**. Optional division by an estimate of
   ``binom(active_count, k)`` to bound magnitudes across position
   density. Followed by LayerNorm.
6. **Delta head**. ``LayerNorm + Linear + GELU + Dropout + Linear``
   on `cat(e, joint)` to a scalar `primitive_delta_raw`.
7. **Gate**. MLP over `cat(joint, tmpp_active_mean, tmpp_coeff_norm)`
   to a sigmoid `primitive_gate`; initial bias `gate_init = -2.0`
   so the primitive starts near-closed.
8. **Output**. ``final_logit = base_logit + primitive_gate *
   primitive_delta_raw``.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full TMPP architecture (default, K=3). |
| `first_order_only` | **Primary falsifier**. Effective K := 1; collapses to DeepSets-style weighted sum pool. |
| `uniform_mask` | ``m_i := 1`` for all squares. Tests whether the chess-rule occupancy mask matters. |
| `shuffle_mask` | In-batch permutation of the occupancy mask. Decouples mask from position. |
| `shuffle_tokens` | Permute token (square) order. Coefficient values are invariant; LayerNorm reductions too. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed by the model. The
operator depends only on the simple_18 piece-presence planes and the
side-to-move plane.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Trunk joint refeat | Two encoder passes total (one for trunk forward, one for joint features) — same as p035 and p001 |
| Polynomial scan | O(64 * K * latent_dim) Hadamard adds |
| Delta head + gate | Small MLPs |

At defaults (``latent_dim=24``, ``K=3``, B=64) the coefficient tensor
is ``(B, 3, 24)`` — under 20 KB per batch. The head adds ~50k
parameters to the trunk.

## Implementation Binding

- Registered model name: `truncated_multiset_polynomial_pool`.
- Source implementation: `src/chess_nn_playground/models/primitives/truncated_multiset_polynomial_pool.py`.
- Shared trunk helper: `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p042_truncated_multiset_polynomial_pool/model.py`.
- Training config: `ideas/registry/p042_truncated_multiset_polynomial_pool/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["truncated_multiset_polynomial_pool"] = build_truncated_multiset_polynomial_pool_from_config`.
