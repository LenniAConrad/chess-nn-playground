# Architecture

`Residual Calibration Error Field` is a board-only `puzzle_binary` classifier
that decomposes prediction into a raw CNN logit and a learned residual
calibration field.

The model accepts the repository board tensor contract `B x 18 x 8 x 8`, adds
rank/file coordinate planes, and encodes the board with a compact CNN. The
pooled CNN feature map produces the baseline score:

`raw_logit = raw_head(pool(features))`.

## Calibration Error Field

A spatial branch predicts an error field from the same intermediate feature map:

`error_field = conv_head(features)`.

The model pools this field and derives two residual calibration terms:

`temperature = softplus(temperature_head(pool(error_field))) + temperature_floor`

`correction = correction_scale * tanh(correction_head(pool(error_field)))`

The final puzzle logit is:

`logits = raw_logit / temperature + correction`.

The bounded additive correction is the trainable residual logit adjustment. The
positive sample-wise temperature is the calibration component: values above one
soften overconfident raw logits, while values below one sharpen underconfident
raw logits.

## Diagnostics

The model returns scalar diagnostics for both the raw and calibrated paths:
`raw_logit`, `calibration_temperature`, `calibration_correction`,
`correction_norm`, `correction_regularizer`, `temperature_log`,
`raw_probability`, `calibrated_probability`, `confidence_delta`, and
`calibration_strength`.

It also returns error-field summaries: `error_field_energy`,
`error_field_peak`, `error_field_l1`, `error_field_entropy`,
`error_field_center_mass`, `error_field_edge_mass`, and
`error_field_signed_mean`. The full `error_field` tensor is included as the
spatial heatmap diagnostic; prediction export ignores non-scalar tensors, while
in-memory analysis can inspect the heatmap directly.

## Output Contract

The primary `logits` tensor has shape `(B,)` and is compatible with the
repository's BCE puzzle-binary trainer.

## Implementation Binding

- Registered model name: `residual_calibration_error_field`
- Source implementation file: `src/chess_nn_playground/models/residual_calibration.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i101_residual_calibration_error_field/model.py`
