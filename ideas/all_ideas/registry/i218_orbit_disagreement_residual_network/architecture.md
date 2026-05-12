# Architecture

## Overview

`Orbit Disagreement Residual Network` treats *disagreement* between safe transform
views of the board as the primary tactical signal rather than enforcing or pooling
invariance.

## Components

- Board encoder: a compact convolutional trunk shared across all views.
- View generator: identity, file flip, rank flip, 180-degree rotation, and a
  color-flip (white/black plane swap) view. All views are exact safe transforms
  of the simple_18 board tensor.
- Latent projection: pooled mean/max board features per view -> latent vector.
- Per-view binary logit head used to compute logit disagreement statistics.
- Orbit pooling: invariant orbit mean, residuals from the orbit mean, per-view
  residual norms, and the residual covariance trace and off-diagonal norm.
- Classifier: pooled board features, orbit mean, residual mean/std, and the
  six-element disagreement summary feed an MLP head.

## Diagnostics returned by the forward pass

- `orbit_mean_norm`, `orbit_residual_mean_norm`, `orbit_residual_max_norm`
- `orbit_covariance_trace`, `orbit_covariance_offdiag_norm`
- `view_logit_disagreement`, `view_logit_range`
- `per_view_logit`, `symmetry_residual`

## Implementation Binding

- Registered model name: `orbit_disagreement_residual_network`
- Source implementation file: `src/chess_nn_playground/models/orbit_disagreement.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i218_orbit_disagreement_residual_network/model.py`
