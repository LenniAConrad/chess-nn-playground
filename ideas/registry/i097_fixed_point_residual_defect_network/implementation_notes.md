# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/fixed_point_residual.py`.
- Idea-local wrapper: `ideas/registry/i097_fixed_point_residual_defect_network/model.py`.
- Registry key: `fixed_point_residual_defect_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.
- Batch candidate: `Fixed-Point Residual Defect Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Ablation modes: `fixed_point` (default), `final_latent_only`, `defect_norm_only`, `single_step`, `untied_residual_blocks`, `random_update_operator`.
