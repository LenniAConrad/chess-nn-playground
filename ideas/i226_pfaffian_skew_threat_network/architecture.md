# Architecture

## Overview

`Pfaffian Skew Threat Network` builds a learned skew-symmetric chess
interaction operator `K = -K^T in R^{2m x 2m}` and classifies puzzle-likeness
from the signed Pfaffian `pf(K)` (oriented enumerator of perfect matchings)
plus a fingerprint of sub-Pfaffians on chess-natural square subsets.

## Components

- Board encoder: convolutional trunk + pooled mean/max summary.
- Upper head: produces the strict upper-triangle of `K`; the model fills the
  lower triangle as `-upper^T` so `K` is skew by construction.
- Pfaffian block: differentiably approximates `pf(K)` via the product of
  positive imaginary parts of the eigenvalues, with an upper-triangle sign
  proxy for orientation tracking.
- Sub-Pfaffian fingerprint: a fixed family of even-sized index subsets
  `(I_q)_q` (deterministic from a seeded permutation) yields per-subset
  signed Pfaffians; their mean sign is the orientation balance score.
- Spectral features: Frobenius norm, spectral norm, nuclear norm, stable
  rank, smallest singular value.
- Classifier: pooled board features + scalar / fingerprint / spectral
  features feed an MLP head.

## Diagnostics returned by the forward pass

- `pfaffian_signed_log`, `pfaffian_log_abs`, `pfaffian_sign`
- `pfaffian_sign_balance`, `pfaffian_subset_signs`,
  `pfaffian_subset_log_abs`
- `pfaffian_frobenius`, `pfaffian_spectral`, `pfaffian_stable_rank`

## Implementation Binding

- Registered model name: `pfaffian_skew_threat_network`
- Source implementation file: `src/chess_nn_playground/models/pfaffian_skew_threat.py`
- Idea-local wrapper: `ideas/i226_pfaffian_skew_threat_network/model.py`
