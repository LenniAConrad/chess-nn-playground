# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/oriented_matroid_covector.py`.
- Idea-local wrapper: `ideas/registry/i096_oriented_matroid_covector_bottleneck/model.py`.
- Registry key: `oriented_matroid_covector_bottleneck`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.
- Batch candidate: `Oriented Matroid Covector Bottleneck`.
- Modes: `covector` (default), `magnitude_only`, `random_hyperplanes`, `material_role_hist_only`, `coordinate_shuffle_by_piece` (selected via `model.mode` in the config).
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
