# Math Thesis

## Working thesis

A true puzzle has a defender block whose effective response, after attacker
influence is algebraically eliminated, is *insolvent*: the Schur complement
`S = D - B^T A^{-1} B` has at least one significantly negative eigenvalue
that no defender configuration can compensate. A near-puzzle has comparable
surface pressure but a near-PSD `S`.

## Setup

Let `M in R^{64 x 64}` be a learned PSD interaction matrix on board squares
and let `m_A in [0, 1]^{64}` be a learned attacker mask. After permuting
attacker squares to the front and partitioning,

```text
M = [ A   B   ]
    [ B^T D   ]
```

with `A in R^{a x a}`, `D in R^{(64 - a) x (64 - a)}`, the Schur complement

```text
S = D - B^T A^{-1} B
```

is the residual defender operator after eliminating `A`. Because `M` is PSD,
`S` is symmetric; its inertia is Haynsworth-additive and
`det(M) = det(A) det(S)`.

## Claim

Soft Haynsworth inertia of `S` (count of negative vs positive eigenvalues
weighted by magnitude) is a near-sufficient statistic for puzzle-likeness
once a basic board pool feature is given.

## Falsifiers

- `no_block_partition`: drop the partition and use the spectrum of full `M`.
- `random_attacker_mask`: randomize `m_A` while preserving `|m_A|`.
- `inertia_only`: keep only soft `(n_pos, n_zero, n_neg)`.
- `zero_off_diagonal`: zero `B`. Then `S = D` and the elimination signal
  vanishes.
