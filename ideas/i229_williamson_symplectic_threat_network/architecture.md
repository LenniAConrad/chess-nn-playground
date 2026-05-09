# Architecture

## Overview

`Williamson Symplectic-Eigenvalue Threat Network` builds a per-board SPD
operator `M in S^{2n}_{++}` over a chess position-momentum phase space
and reads off the symplectic spectrum `{d_i}_{i=1..n}` from Williamson's
normal form `M = S^T D S, S in Sp(2n, R), D = diag(d_1, ..., d_n, d_1, ...,
d_n)`. The puzzle logit is produced from the symplectic spectrum (top-k,
adjacent gaps, entropy, Heisenberg slack) plus the ordinary spectrum of
`M` (for contrast / falsifier `ordinary_eigvals_swap`) and a pooled board
summary.

## Components

- Board encoder: convolutional trunk with mean+max pooling.
- SPD `M` builder: per-board non-negative weights `w_p(X) = softplus(W
  pooled)` weight a fixed bank of learnable PSD primitive factors
  `F_p in R^{2n x r}` (so each `E_p = F_p F_p^T >= 0`); the operator is
  `M(X) = sum_p w_p(X) F_p F_p^T + lambda I_{2n}`, which is SPD by
  construction. Three primitives are seeded toward chess-natural
  position-position, momentum-momentum, and position-momentum couplings.
- Symplectic eigenvalue block: stable algorithm
  - `M^{1/2}` via `eigh`,
  - `K = M^{1/2} J M^{1/2}` (skew-symmetric; `J = [[0, I_n], [-I_n, 0]]`
    is fixed, not learned),
  - eigenvalues of `K^T K` are `{d_i^2}` each with numerical multiplicity
    two,
  - sort descending, average paired entries to dampen multiplicity-2
    splitting, take square roots to obtain `{d_i}`.
- Ordinary spectrum: top-`k` eigenvalues of `M` are returned alongside
  for the `ordinary_eigvals_swap` ablation.
- Head: pooled board features concatenated with
  `[d_topk, eig_topk(M), gaps(d_topk), heisenberg_slack(d_topk),
  symplectic_entropy, log det M, d_min, d_max, heisenberg_violation]`
  feed an MLP that returns one puzzle logit.

## Diagnostics returned by the forward pass

- `symplectic_spectrum`, `symplectic_top_d`, `symplectic_spectral_gaps`
- `symplectic_entropy = -sum_i log d_i`, `symplectic_log_det_M = 2 sum_i
  log d_i`
- `symplectic_d_min`, `symplectic_d_max`
- `heisenberg_slack = d_topk - 1/2`, `heisenberg_violation = sum_i max(0,
  1/2 - d_i)`
- `ordinary_eigvals_topk` (top-`k` eigenvalues of `M`, contrast spectrum)

## Implementation Binding

- Registered model name: `williamson_symplectic_threat_network`
- Source implementation file: `src/chess_nn_playground/models/williamson_symplectic_threat_network.py`
- Idea-local wrapper: `ideas/i229_williamson_symplectic_threat_network/model.py`
