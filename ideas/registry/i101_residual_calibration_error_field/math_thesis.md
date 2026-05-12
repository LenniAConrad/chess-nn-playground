# Math Thesis

Residual Calibration Error Field

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `5`.

Working thesis: If the existing CNN has good accuracy but poor reliability on
near-puzzles, a residual calibration architecture can predict where the baseline
is likely overconfident. The model learns a spatial calibration error field and
uses it to adjust logits or produce diagnostics.

The implemented classifier follows the source packet equation:

`error_field = conv_head(features)`

`temperature = softplus(pool(error_field)) + eps`

`correction = small_mlp(pool(error_field))`

`logit = raw_logit / temperature + correction`

The residual field is not a replacement classifier. It models calibration
residuals of the baseline CNN by predicting a sample-wise temperature and a
bounded additive correction from spatial error evidence.
