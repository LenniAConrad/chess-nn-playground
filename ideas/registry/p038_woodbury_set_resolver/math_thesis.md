# Math Thesis

Source: `ideas/research/primitives/external_33_esp_permanent_woodbury_orbit_primitives.md`,
rank-3 proposal `primitive_woodbury_resolver`. Same primitive as
`primitive_rank1_resolvent_pool` in
`external_36_exterior_product_rank1_resolvent_primitives.md` so both
proposals are covered by this single registry entry.

The rank-1 proposal in the file 33 packet
(`primitive_esp_set`) is the same elementary-symmetric polynomial
operator already implemented as `p024
event_symmetric_interaction_accumulator`, and the rank-4 proposal
(`primitive_orbit_canonicalizer`) duplicates p036, so neither is
re-implemented here.

## Working thesis

For active piece tokens with embeddings `U_i in R^r`, value vectors
`V_i in R^{d_v}`, occupancy mask `m_i in {0, 1}`, and Tikhonov
regulariser `lambda > 0`:

    A = lambda * I + sum_i m_i U_i U_i^T      in R^{r x r}, SPD by construction
    S = sum_i m_i U_i V_i^T                   in R^{r x d_v}
    P = A^{-1}                                in R^{r x r}, SPD
    Y_q = Q_q @ P @ S                          in R^{d_v}
    l_i = U_i^T P U_i                          (per-token leverage)
    s   = log det A                            (capacity term)

The primitive returns `(Y, l, s)`. The defining operation is the
maintained inverse-precision resolver `P`: in an engine make/unmake
loop a piece move adds or removes one rank-one outer-product term, and
the inverse is updated by Sherman-Morrison

    A' = A + s * U_e U_e^T
    P' = P - s * (P U_e)(P U_e)^T / (1 + s U_e^T P U_e)

with `s = +1` for add and `s = -1` for remove, plus the analogous
update for `S`. The training-time path recomputes the static result by
Cholesky-solving the SPD system, which is mathematically equivalent.

## Gradient path

For an SPD `A` and Cholesky factor `L`:

    A = L L^T
    P S = cholesky_solve(S, L)
    l_i = U_i . cholesky_solve(U_i, L)
    log det A = 2 * sum log diag(L)

PyTorch's `torch.linalg.cholesky` and `torch.cholesky_solve` provide
implicit-differentiation gradients

    d (P S)   = -P (dA) (P S) + P (dS)
    d (log det A) = trace(P dA)

so the model layer's parameter gradients flow back through these
identities automatically.

## Architecture-level claim

    final_logit(x) = i193_trunk(x) + sigmoid(g(joint)) * delta(Y, log det A, leverage_mean)

`Y in R^{B, m, d_v}` is the projected queries' resolved output,
`log det A` exposes the "capacity" of the active piece set, and
`leverage_mean` averages per-piece leverage over the occupied squares.
The gate is initialised closed (`gate_init = -2.0`).

## Falsifiers

- Primitive-level: `shuffle_active_tokens` (in-batch permutation of the
  active token tensor) must lose the slice lift.
- `diagonal_only` (zero off-diagonal of `A` before the solve) must lose
  the redundancy-suppression component of the lift -- the inverse
  collapses to per-channel rescaling.
- `uniform_queries` must lose the trunk-conditioned query component.
- Architecture-level: p038 must beat i193 on its declared slice (high
  leverage variance positions, i.e. one piece carrying most of the
  evidence) without regressing aggregate PR AUC.

## Why this is not linear attention

Linear attention pools `phi(q) M / (phi(q) z)` with a positive feature
map; there is no SPD inverse and no leverage scores. The Woodbury
resolver pools through `Q P S` with `P = (lambda I + sum_i U_i U_i^T)^{-1}`,
so collinear tokens cancel rather than add. Leverage scores expose the
direction-specific contribution of each piece, which the cross-correlated
defender pattern in chess directly exploits.

## Why this is not ridge regression

Ridge regression returns `(X^T X + lambda I)^{-1} X^T y`. The Woodbury
resolver pools `U` and `V` independently and applies the inverse to a
cross-covariance, returning a tensor whose gradient passes through the
inverse-precision matrix. The maintained-inverse contract -- exact
rank-one add/delete via Sherman-Morrison -- is the primitive's defining
property, not a generic linear-system solve.
