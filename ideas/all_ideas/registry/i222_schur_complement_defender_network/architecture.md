# Architecture

## Overview

`Schur-Complement Defender Elimination Network` builds a learned PSD board
interaction matrix `M in R^{64 x 64}`, partitions the squares into an attacker
block (top-`a` squares by an attacker score head) and a defender block of size
`64 - a`, and computes the Schur complement `S = D - B^T A^{-1} B`.

## Components

- Board encoder: convolutional trunk with pooled mean/max summary.
- Factor head: produces a 16-rank factor `F in R^{64 x 16}` per board, used to
  form `M = F F^T / 16 + eps I_{64}` (PSD by construction).
- Attacker score head: per-square score from the encoder; the `a` highest-
  scoring squares form the attacker block.
- Block partition + Cholesky solve:
  `Z = chol_solve(A + eps I, B)`, `S = D - B^T Z`.
- Spectral / inertia readout: soft `(n_pos, n_zero, n_neg)`, top-`k`
  eigenvalues of `S`, log-determinants of `S` and `M`, trace, stable rank.
- Classifier: pooled board features + inertia + spectrum + log-determinant
  features feed an MLP head.

## Diagnostics returned by the forward pass

- `schur_inertia_pos`, `schur_inertia_zero`, `schur_inertia_neg`
- `schur_log_det_S`, `schur_log_det_M`, `schur_trace_S`, `schur_stable_rank`
- `schur_eigvals_topk`, `schur_spectral_norm`

## Implementation Binding

- Registered model name: `schur_complement_defender_network`
- Source implementation file: `src/chess_nn_playground/models/schur_complement_defender.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i222_schur_complement_defender_network/model.py`
