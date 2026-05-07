# Architecture

## Overview

`Numerical-Range Boundary Network` builds a learned non-symmetric chess
operator `A in R^{r x r}` and samples its field-of-values `W(A)` boundary
along `K` angles by computing the top eigenvalue of a Hermitian-equivalent
real symmetric block matrix at each angle.

## Components

- Board encoder: convolutional trunk with pooled mean/max summary.
- Operator head: linear projection to `r x r`, spectrally normalized.
- Boundary sampler: at each angle theta, builds the block-real Hermitian
  representation
  `[[cos(t) sym, -sin(t) skew], [sin(t) skew, cos(t) sym]]`
  whose top eigenvalue equals the support of `W(A)` along
  `e^{i theta}`.
- Spectrum block: complex `eigvals(A)`, spectral radius `rho(A)`.
- Non-normality features: numerical radius `numr = max_k mu_k`, gap
  `numr - rho`, Crawford number `min_k mu_k`, boundary curvature.
- Classifier: pooled board features + boundary support + curvature +
  `(gap, numr, rho, crawford, std, mean curvature)` feed an MLP head.

## Diagnostics returned by the forward pass

- `numerical_radius`, `spectral_radius`, `non_normality_gap`
- `crawford_number`, `boundary_support`, `boundary_curvature`
- `boundary_support_std`

## Implementation Binding

- Registered model name: `numerical_range_boundary_network`
- Source implementation file: `src/chess_nn_playground/models/numerical_range_boundary.py`
- Idea-local wrapper: `ideas/i224_numerical_range_boundary_network/model.py`
