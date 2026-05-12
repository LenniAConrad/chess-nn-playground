# Implementation Notes

- Bespoke model implementation: `src/chess_nn_playground/models/trunk/finite_field_character_sum.py`.
- Idea-local wrapper: `ideas/registry/i067_finite_field_character_sum_board_network/model.py` delegates to `build_finite_field_character_sum_board_network_from_config`.
- Registry key: `finite_field_character_sum_board_network` (registered in `src/chess_nn_playground/models/registry.py`).
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2115_friday_shanghai_character_sums.md`.
- This idea is intentionally board-only (`simple_18`) and does not consume engine, verification, source, or CRTK metadata as input.
- Falsifier ablations are exposed via the `ablation` config field: `none`, `residue_only`, `material_polynomial_only`, `random_residue_remap`, `phase_batch_shuffle`, `single_prime`, `real_polynomial_mlp`.
