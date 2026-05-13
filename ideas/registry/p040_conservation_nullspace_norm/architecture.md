# Architecture

`Conservation-Nullspace Normalization` (p040) is an additive, gated
head over the existing i193 trunk. It normalises a per-square latent
after projecting out a fixed rule-derived charge subspace, exposes the
residual norm, charge coefficients, and per-channel sigma, and adds a
gated logit delta.

## Mechanism

1. **i193 trunk forward**. The `ExchangeThenKingDualStreamNetwork`
   runs unchanged.

2. **Per-square latent**. A linear map projects the trunk joint
   feature to `X in R^{B, 64, latent_dim}`.

3. **Per-square weights**. A 1x1 convolution of the simple_18 board
   tensor produces a per-square logit; `softplus` + `weight_bias` keeps
   the weights strictly positive.

4. **Charge matrix `C`**. A fixed `(64, r=8)` buffer encoding the
   eight conservation columns (constant, file, rank, parity, king-zone
   proximity, edge row / col, corner). Built once at construction
   time.

5. **SPD assembly**. `A = C^T D C + epsilon I_r in R^{B, r, r}`, with
   `D = diag(w)`. `b = C^T D X in R^{B, r, d}`.

6. **Cholesky solve**. `L = chol(A)`; `M = cholesky_solve(b, L) in R^{B, r, d}`.

7. **Residual + normalisation**. `R = X - C M`; `sigma^2 = (w * R^2).sum(dim=1) / max(1, sum_w - r)`;
   `Y = gamma * R / sqrt(sigma^2 + epsilon) + beta`.

8. **Pool**. `residual_pool = RMS(Y, dim=64) in R^d`;
   `coeff_pool = mean(|M|, dim=d) in R^r`;
   `sigma_pool = sigma in R^d`.

9. **Readout**. Concat the three pools, pass through LayerNorm +
   Linear + GELU + Dropout + Linear -> `primitive_delta_raw`.

10. **Gate**. Small MLP on the trunk joint feature; `gate_init = -2.0`.

11. **Logit fusion**. `final_logit = base_logit + gate * delta_raw`.

## Ablations

| ID | `model.ablation` | What it tests |
|---|---|---|
| A1 | `shuffle_residual` | In-batch permutation of `Y`. Primary falsifier. |
| A2 | `no_projection` | Set `M = 0`, recover plain weighted normalisation. Tests whether the projection is load-bearing. |
| A3 | `uniform_weights` | Drop per-square weights (`D = I`). Tests whether the rule-derived weight projection is load-bearing. |
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
| Latent projection | `O(feature_dim * 64 * d)` |
| Weight projection | `O(input_channels * 64)` |
| `A` assembly | `O(64 * r^2)` |
| `b` assembly | `O(64 * r * d)` |
| Cholesky + solve | `O(r^3 + r^2 * d)` |
| Residual + normalisation | `O(64 * d)` |
| Readout | `O((d + r + d) + head_hidden_dim)` |

With `d = 16`, `r = 8`: roughly `O(64 * (r^2 + r * d)) = O(64 * (64 + 128)) = O(12288)`
plus the trunk-level latent projection.

## Implementation Binding

- Registered model name: `conservation_nullspace_norm`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/conservation_nullspace_norm.py`.
- Idea-local wrapper:
  `ideas/registry/p040_conservation_nullspace_norm/model.py`.
- Training config:
  `ideas/registry/p040_conservation_nullspace_norm/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`.
