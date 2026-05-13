# Architecture

`Truncated Exterior Product Pool` (p041) is an additive, gated head
over the existing i193 trunk. It pools active piece tokens through the
truncated exterior algebra and projects the grade-decomposed
multivector to a scalar gated delta.

## Mechanism

1. **i193 trunk forward**. The `ExchangeThenKingDualStreamNetwork`
   runs unchanged and emits `logits` (`base_logit`) and the dual-stream
   joint feature.

2. **Per-square token construction**. For each square `s in 0..63` we
   concat the simple_18 piece planes (12 channels) and the side-to-
   move plane into a 13-dimensional token vector.

3. **`z` projection**. A linear map produces `z_i = tanh(W token_i)`
   in `R^r` (default `r = 4`). The `tanh` keeps the multivector
   products bounded.

4. **Wedge tables**. At construction time we precompute
   `target_basis_{k}` and `target_sign_{k}` for each grade `k = 1..R`
   (see `math_thesis.md`). These are static long / float buffers.

5. **Truncated exterior pool**. Starting from `M^{(0)} = 1` and
   `M^{(>=1)} = 0`, iterate over the 64 tokens. For each token `i`
   with mask `a_i`, multiply `M' = M * (1 + a_i z_i)` in the truncated
   algebra by a `scatter_add_` per grade `k = R..1`. Tokens for empty
   squares contribute zero, so they are no-ops.

6. **Readout**. Concatenate `M^{(0)}..M^{(R)}` to a `R^{D_R}` vector
   and pass through LayerNorm + Linear + GELU + Dropout + Linear,
   producing `primitive_delta_raw`.

7. **Gate**. Small MLP on the trunk joint feature with
   `gate_init = -2.0`.

8. **Logit fusion**. `final_logit = base_logit + gate * delta_raw`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_grades_high` | In-batch permutation of `M^{(k)}` for `k >= 2`. Primary falsifier. |
| A2 | `first_order_only` | Zero `M^{(>=2)}`. Tests whether higher-grade wedge cancellation is load-bearing. |
| A3 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A4 | `trunk_only` | Alias of `zero_delta`. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass through the dual-stream encoder |
| Token projection | `O(64 * 13 * r)` |
| Exterior pool | `O(64 * sum_k (D_{k-1} * r))` per active token, dominated by the highest grade. For `r = 4, R = 3`: 64 * (1*4 + 4*4 + 6*4) = 64 * 44 = 2816 scatter contributions. |
| Readout | `O(D_R + head_hidden_dim)` |

`D_R = 1 + r + C(r, 2) + C(r, 3) = 1 + 4 + 6 + 4 = 15` for the
default `r = 4`, `R = 3`. The wedge update is the operator's main
cost and stays well within scout-scale budgets at this size.

## Implementation Binding

- Registered model name: `truncated_exterior_product_pool`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/truncated_exterior_product_pool.py`.
- Idea-local wrapper:
  `ideas/registry/p041_truncated_exterior_product_pool/model.py`.
- Training config:
  `ideas/registry/p041_truncated_exterior_product_pool/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.
