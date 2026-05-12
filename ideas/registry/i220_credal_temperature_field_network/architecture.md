# Architecture

## Overview

`Credal Temperature Field Network` keeps the standard binary puzzle classifier
but adds a sample-wise calibration branch that predicts a positive temperature
`T(x)` and bounded smoothing `alpha(x)` from current-board features.

## Components

- Board encoder: convolutional trunk + pooled mean/max board summary.
- Shared MLP head (LayerNorm + Linear + GELU + Dropout).
- Logit head produces the raw puzzle logit `z`.
- Temperature head produces `T(x) = softplus(t(x)) + temperature_floor`.
- Smoothing head produces `alpha(x) = max_alpha * sigmoid(a(x))`.
- Final logit: `(1 - alpha(x)) * z / T(x)`.

## Diagnostics returned by the forward pass

- `raw_logits`, `calibrated_logits`
- `credal_temperature`, `credal_temperature_log`
- `credal_smoothing`, `credal_entropy`

## Implementation Binding

- Registered model name: `credal_temperature_field_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/credal_temperature.py`
- Idea-local wrapper: `ideas/registry/i220_credal_temperature_field_network/model.py`
