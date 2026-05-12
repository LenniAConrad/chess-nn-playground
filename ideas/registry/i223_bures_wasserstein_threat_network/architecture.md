# Architecture

## Overview

`Bures-Wasserstein SPD Threat Manifold Network` embeds each board into the cone
of symmetric positive-definite matrices via a learned threat covariance and
classifies puzzle-likeness using the Bures-Wasserstein metric on `S^d_{++}`.
The model uses operator-geometric-mean closed forms rather than Fisher-Rao or
log-Euclidean geometry.

## Components

- Board encoder: convolutional trunk plus pooled mean/max board summary.
- Feature projection: `1 x 1` conv to a `(B, d, 8, 8)` field, then
  `Sigma = (1 / 64) F^T F + eps I_d`.
- Class Frechet means: two learnable Cholesky factors for `mu_0`, `mu_1`,
  reprojected to the SPD cone as `L L^T + eps I_d`.
- Bures distance: differentiable
  `d_BW(Sigma, mu)^2 = tr(Sigma) + tr(mu) - 2 tr[(Sigma^{1/2} mu Sigma^{1/2})^{1/2}]`
  using `eigh`-based symmetric square roots.
- Tangent log map: `T_{mu -> Sigma} - I` with
  `T = mu^{-1/2} (mu^{1/2} Sigma mu^{1/2})^{1/2} mu^{-1/2}`, then upper-
  triangle vectorization.
- Classifier: pooled board features + per-class log-map features +
  `(d_BW_0, d_BW_1, d_BW_0 - d_BW_1)` + `log|det Sigma|` + `tr(Sigma)`
  feed an MLP head.

## Diagnostics returned by the forward pass

- `bures_distance_class0`, `bures_distance_class1`, `bures_distance_gap`
- `bures_log_det_sigma`, `bures_trace_sigma`, `bures_spectral_sigma`
- `bures_log_phi0`, `bures_log_phi1`

## Implementation Binding

- Registered model name: `bures_wasserstein_threat_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/bures_wasserstein_threat.py`
- Idea-local wrapper: `ideas/registry/i223_bures_wasserstein_threat_network/model.py`
