# Architecture

`Woodbury Set Resolver` (p038) is an additive, gated head over the
existing i193 trunk. It maintains an active-set inverse-precision
memory `P = (lambda I + sum_i U_i U_i^T)^{-1}` and applies it to
trunk-derived queries plus a value cross-covariance.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit plus a per-sample
diagnostic dict.

## Mechanism

1. **i193 trunk forward**. The `ExchangeThenKingDualStreamNetwork`
   runs unchanged and emits `logits` (`base_logit`) and the dual-stream
   joint feature.

2. **Per-square token construction**. For each square `s in 0..63` we
   concat the simple_18 piece planes (12 channels), side-to-move (1
   channel), and four castling-rights channels into a 17-dimensional
   token vector, then mask by `occupancy[s]`.

3. **`U`/`V` projections**. Two linear maps produce
   `U_i in R^r` (default `r = u_dim = 12`) and `V_i in R^{d_v}` (default
   `d_v = v_dim = 16`).

4. **SPD precision `A`**. `A = lambda I + sum_i U_i U_i^T`, in
   `R^{B, r, r}`. The Tikhonov regulariser `lambda` (default `1e-2`)
   keeps `A` strictly positive-definite.

5. **Cross-covariance `S`**. `S = sum_i U_i V_i^T`, in `R^{B, r, d_v}`.

6. **Queries**. The trunk joint feature is linearly projected to
   `Q in R^{B, m, r}` (default `m = num_queries = 4`).

7. **Cholesky solve**. `L = chol(A)`; `P S = cholesky_solve(S, L)`;
   `Y = Q @ P S` in `R^{B, m, d_v}`. Per-token leverage
   `l_i = U_i^T P U_i` is computed by one additional Cholesky solve.

8. **Log-determinant**. `log det A = 2 * sum log diag(L)`.

9. **Readout**. Concatenate `Y.flatten()`, `log det A`, and the
   leverage mean over active pieces into a
   `R^{m * d_v + 2}` readout vector. Run through a LayerNorm + Linear +
   GELU + Dropout + Linear stack producing `primitive_delta_raw`.

10. **Gate**. A second small MLP on the trunk joint feature produces
    `gate_logit`. The effective gate is `sigmoid(gate_logit)`,
    initialised near zero by `gate_init = -2.0`.

11. **Logit fusion**. `final_logit = base_logit + gate * delta_raw`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_active_tokens` | In-batch permutation of the active token tensor. Primary falsifier. |
| A2 | `diagonal_only` | Zero off-diagonal of `A` before the solve. Tests whether off-diagonal coupling (the inverse-precision redundancy suppression) is load-bearing. |
| A3 | `uniform_queries` | Replace `Q` with all-ones (matched dim). Tests trunk-conditioned query routing. |
| A4 | `zero_delta` | Hold `primitive_delta = 0`. Recovers i193. |
| A5 | `trunk_only` | Alias of `zero_delta`. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed by the model.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk forward | One pass through the dual-stream encoder |
| `U`/`V` projection | `O(64 * token_input_dim * (r + d_v))` |
| `A` assembly | `O(64 * r^2)` |
| `S` assembly | `O(64 * r * d_v)` |
| Cholesky | `O(r^3)` |
| Two Cholesky solves | `O(r^2 * (d_v + 64))` |
| Readout | `O((m * d_v + 2) + head_hidden_dim)` |

With default `r = 12`, `d_v = 16`, `m = 4` and 64 squares, this
sums to roughly `O(64 * (r^2 + r * d_v))` per sample, comparable to
one extra linear layer at trunk width.

## Implementation Binding

- Registered model name: `woodbury_set_resolver`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/woodbury_set_resolver.py`.
- Idea-local wrapper:
  `ideas/registry/p038_woodbury_set_resolver/model.py`.
- Training config:
  `ideas/registry/p038_woodbury_set_resolver/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.
