# Math Thesis

## Working thesis

Tactical pressure on the board is non-normal: attacker influence flowing
through ray geometry with asymmetric defender resistance produces operators
that *amplify transiently* before the spectrum shows it. The numerical range

```text
W(A) = { x* A x : x in C^n, ||x|| = 1 }    subset C
```

has

```text
rho(A) <= numr(A) <= ||A||_2 <= 2 numr(A),
```

and `numr(A) - rho(A)` measures transient amplification / pseudospectral
spread that no spectrum-only readout sees.

## Setup

Build a low-rank operator `A in R^{r x r}` from the board encoder. For each
angle `theta_k in [0, pi)`, the support of `W(A)` along `e^{i theta_k}` is
the top eigenvalue of the Hermitian
`H_k = (e^{-i theta} A + e^{i theta} A^*) / 2`. We compute it via a
real-only block representation
`[[cos sym, -sin skew], [sin skew, cos sym]]` whose top eigenvalue equals
the same support function. The boundary support sequence, its curvature,
the numerical radius, the spectral radius of `A`, and the gap
`numr - rho` form the discriminative features.

## Claim

The non-normality gap `numr(A) - rho(A)` correlates with puzzle-likeness
*after* controlling for board-pool features.

## Falsifiers

- `force_normal_A`: project `A` onto normal matrices.
- `gap_only_scalar`: keep only the scalar gap.
- `boundary_only_no_spec`: remove `rho`.
