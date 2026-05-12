# Implementation Notes

- Central code: `src/chess_nn_playground/models/baseline_logit_residual_adapter.py`.
- Idea-local wrapper: `ideas/registry/i098_baseline_logit_residual_adapter/model.py`.
- Registry key: `baseline_logit_residual_adapter` (registered directly in `MODEL_BUILDERS`).
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.
- Batch candidate: `Baseline Logit Residual Adapter`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Configurable knobs: `channels`, `hidden_dim`, `depth`, `dropout`, `use_batchnorm`, `adapter_channels` (defaults to `max(8, channels // 2)`), `residual_scale` (default `1.0` — set to `0.0` to recover the baseline-only ablation), and `detach_baseline_context` (default `True`).
