# Math Thesis

## Working thesis

Attacker `A` and defender `B` are *approximately freely independent* under
non-puzzle conditions, so the spectrum of `A + B` matches the
free-additive-convolution prediction `mu_A boxplus mu_B`. A puzzle is the
breaking of free independence: the empirical spectrum of `A + B` deviates
from the free-convolution prediction in a way no classical (commuting)
spectrum analysis can detect.

## Setup

Build `A, B in R^{n x n}`, symmetrize to get Hermitian `A_sym`, `B_sym`,
and spectrally normalize. Compute eigendecompositions of `A_sym`, `B_sym`,
`A_sym + B_sym`. The first four free cumulants are recovered from raw
moments via the closed-form non-crossing-partition recursion

```text
kappa_1 = m_1
kappa_2 = m_2 - m_1^2
kappa_3 = m_3 - 3 m_1 m_2 + 2 m_1^3
kappa_4 = m_4 - 4 m_1 m_3 - 2 m_2^2 + 10 m_1^2 m_2 - 5 m_1^4.
```

Free additive convolution sets `kappa_pred = kappa_A + kappa_B`. We
approximate the predicted spectral measure by a Gaussian sharing the first
two predicted cumulants and compute the Wasserstein-1d distance to the
empirical spectrum of `A_sym + B_sym` as the coupling distance.

## Claim

The free-cumulant mismatch
`kappa_k(A + B) - kappa_k(A) - kappa_k(B)` is non-zero precisely on tactical
positions and approximately zero on non-tactical ones.

## Falsifiers

- `classical_swap`: replace free convolution with `sort(spec(A) + spec(B))`.
- `kappa_only`: drop the Wasserstein distance, keep only the cumulant
  mismatch.
- `B_eq_zero`: should collapse the prediction to `A`.
