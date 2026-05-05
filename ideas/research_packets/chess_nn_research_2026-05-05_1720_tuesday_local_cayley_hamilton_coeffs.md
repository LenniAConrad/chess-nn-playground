# Codex Research Packet: Cayley-Hamilton Coefficient Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1720_tuesday_local_cayley_hamilton_coeffs.md`
- Generated at: 2026-05-05 17:20
- Author: Claude (Opus 4.7, 1M context)
- Status: bespoke implementation already in `src/chess_nn_playground/models/cayley_hamilton_coeffs.py`

## Thesis

Extract the **characteristic polynomial coefficients** of a learned `r x r`
chess operator `A` via the **Faddeev-LeVerrier recursion** (no eigendecomp):

```text
M_0 = I
for k in 1..r:
    c_k = -trace(A * M_{k-1}) / k
    M_k = A * M_{k-1} + c_k * I
char_poly(A; lambda) = lambda^r + c_1 lambda^{r-1} + ... + c_r
c_k = (-1)^k * e_k(spec A)         (signed elementary symmetric polynomials of eigenvalues)
```

The vector `(c_1, ..., c_r)` is a power-sum-equivalent encoding of the spectrum
that is **combinatorially distinct** from the eigenvalues themselves: each `c_k`
sums products of `k` eigenvalues — capturing k-element subset interactions of
the spectrum. Faddeev-LeVerrier is fully differentiable and avoids the
gradient-instability of `eigh` near degenerate spectra.

We also compute the Cayley-Hamilton residual

```text
R = A^r + c_1 A^{r-1} + ... + c_r I    (= 0 by Cayley-Hamilton theorem)
```

as a sanity feature and auxiliary loss target.

## Distinct From

- Hessian/operator spectrum (i062, i076, i077, i078, i199, i228): use eigvals or moments directly.
- Tucker certificate (i090): tensor-mode rank, not characteristic polynomial.
- Determinantal volume (i058): only `det = (-1)^r c_r`; we read all r coefficients.

## Architecture

`CayleyHamiltonCoefficientNetwork` in `src/chess_nn_playground/models/cayley_hamilton_coeffs.py`:

```text
input (B, 18, 8, 8)
  -> BoardConvStem -> (B, C, 8, 8) -> pooled (B, C)
  -> Linear -> A in R^{r x r}, spectral-clipped Frobenius <= 1
  -> Faddeev-LeVerrier loop (r matmuls, fully autograd-friendly)
     -> coefficients (c_1, ..., c_r)
  -> features:
       log|c_k|, smooth_sign(c_k)              (2r)
       det(A) = (-1)^r c_r
       trace(A) = -c_1
       sum log|c_k|                              (nuclear-like proxy)
       ||A^r + sum c_k A^{r-k}||_F               (Cayley-Hamilton residual; should be ~0)
  -> concat pooled
  -> MLP -> (B, num_classes)
```

## Ablations

| Ablation | Target |
|---|---|
| `eigvals_swap` | use eigvals(A) instead of (c_k) | tests c_k vs eigvals representation |
| `magnitude_only` | drop signs of c_k | tests sign info |
| `det_only` | use only c_r (= ±det) | tests sufficiency |
| `random_A` | random A unrelated to chess | tests learned A |
| `cnn_same_params` | matched baseline | |

## Falsifier

`eigvals_swap` should NOT drop PR AUC much if the spectrum-as-multiset is sufficient. If `eigvals_swap` *does* drop noticeably, the c_k carry combinatorial structure beyond the eigenvalues' multiset — a positive result. Either outcome is informative.

## Targets

PR AUC ≥ 0.82, F1 ≥ 0.76, near-puzzle FPR ≤ 0.20.
