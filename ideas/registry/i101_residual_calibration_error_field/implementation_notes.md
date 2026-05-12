# Implementation Notes

- Central code: `src/chess_nn_playground/models/residual_calibration.py`.
- Registry key: `residual_calibration_error_field`.
- Idea wrapper: `ideas/registry/i101_residual_calibration_error_field/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.
- Batch candidate: `Residual Calibration Error Field`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The implementation uses a compact CNN baseline logit plus a spatial calibration
  error-field branch that predicts sample-wise temperature and bounded additive
  correction.
