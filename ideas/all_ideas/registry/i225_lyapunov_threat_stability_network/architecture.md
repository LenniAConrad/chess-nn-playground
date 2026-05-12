# Architecture

## Overview

`Lyapunov Stability Threat Network` treats each board as an autonomous linear
system `dot x = A x` with a learned damping that keeps `A` Hurwitz at
initialization, builds a board-derived PSD weighting `Q`, and solves the
continuous Lyapunov equation `A^T P + P A = -Q` via the vec form.

## Components

- Board encoder: convolutional trunk + pooled mean/max summary.
- Flow operator: `A = -damping * I + flow(X_sq)` in `R^{r x r}`.
- Hurwitz clip: subtract `(max_real + safety) * I` from `A` to guarantee a
  unique solution at solve time.
- `Q` head: `Q = factor factor^T + q_floor I` (PSD).
- Lyapunov solver: vec-form solve
  `(I kron A^T + A^T kron I) vec(P) = -vec(Q)` via batched Kronecker.
- Diagnostics: soft Haynsworth inertia of `P`, top-`k` eigenvalues, trace,
  log-determinant, condition number, worst-direction settling proxy,
  Hurwitz indicator on `max real eig(A)`.
- Classifier: pooled board features + inertia + top eigenvalues + scalar
  features feed an MLP head.

## Diagnostics returned by the forward pass

- `lyapunov_inertia_pos`, `lyapunov_inertia_zero`, `lyapunov_inertia_neg`
- `lyapunov_log_det_P`, `lyapunov_trace_P`, `lyapunov_cond_P`
- `lyapunov_spectral_P`, `lyapunov_hurwitz_indicator`,
  `lyapunov_max_real_A`, `lyapunov_eigvals_topk`

## Implementation Binding

- Registered model name: `lyapunov_threat_stability_network`
- Source implementation file: `src/chess_nn_playground/models/lyapunov_threat_stability.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i225_lyapunov_threat_stability_network/model.py`
