# Math Thesis

Matrix-Pencil Generalized Spectrum Bottleneck

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2101_friday_shanghai_matrix_pencil.md`.

Working thesis: puzzle-like positions are characterized by directional
dominance between two learned PSD board-energy forms ``A(x)`` and
``B(x)`` built from current-board occupied tokens. The generalized
eigenvalues of the matrix pencil ``(A, B)``, solving ``A v = lambda B v``,
test that relative spectral geometry more directly than the separate
covariance spectra of either matrix alone.

## Construction

Let ``S(x) = {(t_i, s_i)}_{i=1}^N`` be the occupied piece tokens of board
``x`` (``N <= 32``). A token encoder produces ``h_i in R^D``. The model
constructs two low-rank PSD matrices

```
U_r(x) in R^{K x M}    (r in {A, B}, K = factor_rank, M = matrix_dim)
A(x) = U_A^T U_A / K + eps * I_M
B(x) = U_B^T U_B / K + eps * I_M
```

with ``U_r[k, :] = sum_i weight_r[i, k] * value_r[i, :]`` where
``weight_r`` is a masked softmax over occupied tokens and ``value_r`` is
a per-token MLP output. The regulariser ``eps * I`` keeps ``B`` positive
definite.

## Generalized Eigenproblem

The generalized eigenvalues ``lambda_j(x)`` solve

```
A(x) v_j = lambda_j(x) B(x) v_j.
```

Because ``B`` is positive definite after regularization, with Cholesky
``B = L L^T`` the spectrum is computed as

```
C(x) = L(x)^{-1} A(x) L(x)^{-T}
lambda(x) = eigvals(C_sym(x))      where C_sym = 0.5 * (C + C^T).
```

By Lagrange stationarity the generalized eigenvalues are the extremal
values of the generalized Rayleigh quotient

```
R_x(v) = (v^T A(x) v) / (v^T B(x) v)
```

restricted to ``v^T B v = 1``. The implementation also exposes
``R_x(z_p)`` along ``probe_count`` learned probe directions.

## Falsifiers Implemented

The bespoke model exposes the markdown's section-9 falsifiers via the
``ablation`` config flag: ``separate_spectra_only`` (central falsifier),
``trace_ratio_only``, ``batch_shuffled_b``, ``random_factors``,
``single_matrix_spectrum``, ``mean_pool_head``, and
``material_only_tokens``. The relative pencil geometry must beat the
``separate_spectra_only`` and ``batch_shuffled_b`` controls before the
result can be attributed to the matrix pencil.

## Implementation Status

This idea is implemented as a bespoke architecture in
`src/chess_nn_playground/models/matrix_pencil_generalized_spectrum_bottleneck.py`,
registered as `matrix_pencil_generalized_spectrum_bottleneck`, and
wrapped by `ideas/registry/i062_matrix_pencil_generalized_spectrum_bottleneck/model.py`.
It is no longer a `ResearchPacketProbe` scaffold.
