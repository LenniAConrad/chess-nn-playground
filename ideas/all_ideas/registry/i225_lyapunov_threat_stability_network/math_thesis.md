# Math Thesis

## Working thesis

A puzzle position corresponds to dynamics that are *not* asymptotically
stable for the defender's natural reaction direction; a non-puzzle has
`Re(spec(A)) << 0` and a well-conditioned Lyapunov certificate `P`.

## Setup

Build an operator `A = -alpha I_r + flow(X_sq)` with `alpha > 0` learnable
and `flow` derived from board features, plus a PSD weighting
`Q = factor factor^T + beta I`. The Lyapunov equation

```text
A^T P + P A = -Q
```

has a unique symmetric solution `P` whenever
`spec(A) cap (-spec(A)) = empty`; if every eigenvalue of `A` has negative
real part, `P` is SPD and `V(x) = x^T P x` is a quadratic stability
certificate. We compute `P` via the vec form
`(I kron A^T + A^T kron I) vec(P) = -vec(Q)` after a small Hurwitz-safety
shift.

## Claim

The condition number, smallest eigenvalue, and soft Haynsworth inertia of
`P` are near-sufficient statistics for puzzle-likeness given the board pool.

## Falsifiers

- `Q_eq_I`: replace chess `Q` with identity.
- `symmetric_A`: force `A <- (A + A^T)/2`. Then `P` is trivial in the
  symmetric `A` case.
- `inertia_only`: drop magnitudes.
