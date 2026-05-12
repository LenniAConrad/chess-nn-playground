# Architecture

## Overview

`p-adic Ultrametric Threat Embedding Network` maps each square through a
learned p-adic encoder `phi(s) in {0,...,p-1}^k`, computes soft ultrametric
distances `d_p(s, t) = p^{-prefix_len(s, t)}`, builds a valuation-weighted
`64 x 64` matrix `M_p` from learned p-adic relation classes, and classifies
puzzle-likeness from the depth histogram of `D`, `M_p` spectrum, and
log-magnitude (Newton-polygon style) slopes of the eigenvalues.

## Components

- Board encoder: convolutional trunk + pooled mean/max summary.
- Digit head: per-square soft digit distribution
  `phi in R^{64 x k x p}` via softmax.
- Prefix-match probability: `match[s, t, i] = sum_d phi_i(s, d) phi_i(t, d)`.
- Cumulative prefix prob and expected divergence depth
  `E[min_diff] = sum_i (1 - prod_{j <= i} match_j)`.
- Ultrametric distance: `D = p^{-E[min_diff]}`.
- Relation head + p-adic absorption: per-square soft relation distribution
  combined into a `R^{64 x 64}` matrix `M_p` whose entries are valuation-
  weighted sums `sum_i p^{-i} f_i(phi_i(K_p))`.
- Symmetric `M_p` -> `eigh` -> top-k eigenvalues by magnitude and Newton-
  polygon slope proxies (consecutive log-magnitude differences).
- Classifier: pooled board features + depth histogram + eigenvalue topk +
  slopes + `D`, `M_p` norm features feed an MLP head.

## Diagnostics returned by the forward pass

- `padic_depth_histogram`, `padic_distance_mean`, `padic_distance_max`
- `padic_spectrum_topk`, `padic_newton_slopes`
- `padic_M_norm`, `padic_spectral_norm`

## Implementation Binding

- Registered model name: `padic_ultrametric_threat_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/padic_ultrametric_threat.py`
- Idea-local wrapper: `ideas/registry/i227_padic_ultrametric_threat_network/model.py`
