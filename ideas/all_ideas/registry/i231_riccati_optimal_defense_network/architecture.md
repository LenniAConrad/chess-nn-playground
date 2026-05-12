# Architecture

## Overview

`Riccati Optimal-Defense Network` builds a per-board low-rank LQR
quadruple `(A, B, Q, R)` of dimensions `A in R^{r x r}, B in R^{r x m},
Q in S^r_+, R in S^m_{++}` (defaults `r = 12`, `m = 4`) from a compact
convolutional trunk and solves the continuous algebraic Riccati equation

```text
A^T P + P A - P B R^{-1} B^T P + Q = 0
```

via the Hamiltonian matrix

```text
H = [[ A,        -B R^{-1} B^T ],
     [ -Q,       -A^T          ]]   in R^{2r x 2r}.
```

The stabilizing solution is read off from the stable invariant subspace
of `H`: take the `r` eigenvectors of `H` whose eigenvalues have the
smallest real part (most stable), partition them as
`V_stable = [V_1; V_2]`, and recover

```text
P = V_2 V_1^{-1}
```

(symmetrized, real part). The puzzle logit is produced from the spectrum
of `P`, the optimal-defense cost `J* = trace(P)`, the optimal feedback
gain `K = R^{-1} B^T P`, the closed-loop spectrum of `A_cl = A - B K`,
and the Hamiltonian's near-imaginary mode count.

## Components

- Board encoder: convolutional trunk with mean+max pooling.
- `A` builder: gated sum of `r x r` primitives biased toward Hurwitz
  via `A = -softplus(alpha) I + tanh-gated sum_p primitives_A[p]`,
  followed by a Hurwitz-safety clip that subtracts
  `max(0, max_real(eig(A)) + safety) * I` so `A` stays strictly stable.
- `B` builder: linear-gated sum of `r x m` primitives.
- `Q` builder: PSD by construction via `Q = sum_p w_p F_p F_p^T + beta I`
  with non-negative gates `w_p = softplus(...)` and a positive floor
  `beta`.
- `R` builder: PD by construction via `R = sum_p w_p G_p G_p^T + gamma I`
  with `gamma > 0`.
- Hamiltonian assembly: the `2r x 2r` block matrix `H` above.
- CARE solver: complex eigendecomposition of `H`, sort eigenvalues by
  real part, take the `r` most-stable eigenvectors as the stable
  invariant subspace, solve `V_1^T X^T = V_2^T` for `X = P` (with a
  small Tikhonov regularizer on `V_1^T`), keep the real symmetric part.
  Differentiation flows through `torch.linalg.eig` and
  `torch.linalg.solve`.
- Diagnostics block: spectrum of `P`, `J* = trace(P)`, `log|det P|`,
  optimal gain `K = R^{-1} B^T P`, closed-loop spectrum
  `spec(A_cl) = spec(A - B K)`, Hamiltonian eigenvalue collection,
  near-imaginary count, CARE residual
  `||A^T P + P A - P B R^{-1} B^T P + Q||_F`.
- Head: pooled board features concatenated with
  `[eig_topk(P), trace(P), log|det P|, J*, ||K||_F, CARE_residual,
  spec_topk(A_cl), min/max Re A_cl, hamiltonian_imag_count,
  hurwitz_indicator]` feed an MLP that returns one puzzle logit.

## Diagnostics returned by the forward pass

- `riccati_eigvals_P`, `riccati_top_eig_P`, `riccati_trace_P`,
  `riccati_log_det_P`
- `riccati_optimal_cost_J_star = trace(P)`, `riccati_gain_norm_K`
- `riccati_closed_loop_top_real`, `riccati_closed_loop_min_real`,
  `riccati_closed_loop_max_real`, `riccati_hurwitz_indicator`
- `riccati_hamiltonian_imag_count`, `riccati_care_residual_F`,
  `riccati_open_loop_max_real_A`

## Implementation Binding

- Registered model name: `riccati_optimal_defense_network`
- Source implementation file: `src/chess_nn_playground/models/riccati_optimal_defense_network.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i231_riccati_optimal_defense_network/model.py`
