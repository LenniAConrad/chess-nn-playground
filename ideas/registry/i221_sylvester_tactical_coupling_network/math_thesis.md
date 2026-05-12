# Math Thesis

## Working thesis

A tactical puzzle position creates a near-resonance between attacker spectrum
and defender spectrum: there is at least one pair `(i, j)` with
`lambda_i(A) + mu_j(B)` close to zero, which inflates the unique solution `X`
of `A X + X B = C` along the matching attacker-defender mode pair.

## Setup

Let `A, B in R^{r x r}` be low-rank attacker/defender operators (spectrally
normalized) and let `C in R^{r x r}` be a board-derived obligation matrix.
The Sylvester equation has a unique solution `X` whenever
`spec(A) cap spec(-B) = empty`, recoverable via the vec form

```text
(I_r kron A + B^T kron I_r) vec(X) = vec(C).
```

The spectral readout of `X` (top-`k` singular values, Frobenius / spectral
norms, soft rank, attacker / defender projected energies, log-volume) and the
cross-spectrum resonance `min_{i,j} |lambda_i(A) + mu_j(B)|` give the
discriminative features.

## Claim

If puzzle structure is genuinely a coupling phenomenon between attacker and
defender linear actions, then puzzle vs near-puzzle separation should be
better with the Sylvester readout than with any single-operator spectrum.

## Falsifiers

- `independent_operators_only`: replace the Sylvester solve with `[A; B]`
  features.
- `swap_AB`: swap attacker and defender slots.
- `static_resonance_only`: keep only `min |lambda + mu|` as a scalar.
- `rank_one_C`: replace `C` by a rank-1 outer product.
