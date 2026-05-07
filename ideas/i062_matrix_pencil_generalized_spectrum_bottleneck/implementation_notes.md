# Implementation Notes

- Central code: `src/chess_nn_playground/models/matrix_pencil_generalized_spectrum_bottleneck.py`.
- Registry key: `matrix_pencil_generalized_spectrum_bottleneck`.
- Idea-local wrapper: `ideas/i062_matrix_pencil_generalized_spectrum_bottleneck/model.py`
  exposes `build_model_from_config(config)` and forwards to
  `build_matrix_pencil_generalized_spectrum_bottleneck_from_config`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2101_friday_shanghai_matrix_pencil.md`.
- Section-9 falsifier ablations exposed via `model.ablation`:
  `separate_spectra_only` (central), `trace_ratio_only`,
  `batch_shuffled_b`, `random_factors`, `single_matrix_spectrum`,
  `mean_pool_head`, and `material_only_tokens`.
- This idea is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
