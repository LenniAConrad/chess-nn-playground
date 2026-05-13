# Math Thesis

Source: `ideas/research/primitives/external_35_espa_conservation_isotypic_green_primitives.md`,
rank-2 proposal `primitive_conserve_norm`. The rank-1 proposal
(`primitive_espa`) duplicates `p024 event_symmetric_interaction_accumulator`
and is not re-implemented.

## Working thesis

Given a per-square latent `X in R^{B, 64, d}`, positive per-square
weights `w in R_+^{B, 64}` with `D = diag(w)`, a *fixed* charge matrix
`C in R^{64, r}`, and an SPD regulariser `epsilon > 0`:

    A = C^T D C + epsilon I_r       in R^{B, r, r}, SPD
    b = C^T D X                       in R^{B, r, d}
    M = A^{-1} b                      in R^{B, r, d}   (weighted least-squares coefficients)
    R = X - C M                       in R^{B, 64, d}  (nullspace residual)
    sigma_j^2 = R[:, j]^T D R[:, j] / max(1, sum_i w_i - r)   in R^{B, d}
    Y[i, j]   = gamma_j * R[i, j] / sqrt(sigma_j^2 + epsilon) + beta_j

`M` is the weighted least-squares projection of `X` onto the column
space of `C`; `R` is the residual after that projection is removed.
The residual is then normalised channel-wise with the unbiased
`max(1, sum w - r)` denominator.

## Charge columns

The charge matrix has 8 fixed columns:

| j | column meaning |
|---|---|
| 0 | constant intercept |
| 1 | file index linearly mapped to [-1, 1] |
| 2 | rank index linearly mapped to [-1, 1] |
| 3 | square parity centred to {-1, +1} |
| 4 | king-zone proximity proxy (1 / (chebyshev(s, e4) + 1)) |
| 5 | edge row indicator (row in {0, 7}) |
| 6 | edge col indicator (col in {0, 7}) |
| 7 | corner indicator |

These are all board-geometry conservation charges. They cover the
"easy" explanations of the input latent: side-to-move-dependent file
preference, central-square bonus, edge / corner penalties, and a
king-zone bias.

If a future variant needs material counts as additional charges
(white pawn count, black pawn count, etc.) they can be appended to `C`
without changing the operator -- but they would need to be computed
per-position rather than as fixed columns, so they are deferred.

## SPD via Cholesky

`A = C^T D C + epsilon I_r` is SPD because `D > 0` and `epsilon > 0`,
so `C^T D C` is PSD and the regulariser strictly bumps it positive.
Cholesky factorisation + `cholesky_solve` provides the gradient through
the inverse via implicit differentiation.

## Architecture-level claim

    final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * delta(residual_pool, M_pool, sigma_pool)

with all three pools computed from the normalised residual `Y`, the
charge coefficients `M`, and the per-channel sigma `sigma`. The gate
is initialised closed (`gate_init = -2.0`).

## Falsifiers

- Primitive-level: `shuffle_residual` (in-batch permutation of `Y`)
  must lose the slice lift.
- `no_projection` (set `M = 0`, recover plain weighted normalisation)
  must lose the conservation-specific lift.
- `uniform_weights` (drop per-square weights, set `D = I`) must lose
  the weighted-normalisation component.
- Architecture-level: p040 must beat i193 on its declared slice
  (positions where the conservation projection explains > median
  fraction of the latent) without regressing aggregate PR AUC.

## Why this is not LayerNorm

LayerNorm normalises by the empirical mean and variance over the
feature dimension, treating all positions equivalently. ConserveNorm
*projects out a specific subspace* (the conservation charges), then
normalises the residual under per-position weights. The normalisation
axis is therefore "what's left after the conservation bookkeeping is
removed", not "the activation distribution".

## Why this is not BatchNorm

BatchNorm normalises across the batch dimension. ConserveNorm
normalises across the spatial (per-square) dimension, with per-sample
weights and a per-sample residual. Two samples in the same batch have
independent normalisation statistics.
