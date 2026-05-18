# Trainer Notes

Use the guarded idea `train.py`. The config is paper-grade, CUDA-required, and uses the canonical tagged CRTK split. New runs must pass `scripts/validate_run_artifacts.py`.

The model loads via the `commutative_view_consistency_network` registry key and lives at `src/chess_nn_playground/models/trunk/commutative_view_consistency.py`. Each forward pass runs the simple_18 board through five view encoders (one CNN + DeepSets + three MLPs) and applies eight rank-`map_rank` cross-view maps, so the per-step parameter count grows with `map_rank * latent_dim` rather than with `latent_dim^2`. The defect statistics tensor is `(batch, 9, 5)` and is fused with the five projected view latents before the LayerNorm + GELU MLP head.

For ablation runs, set `model.ablation` to one of the documented modes (`none`, `views_only_no_defects`, `single_square_view`, `random_view_maps`, `count_to_all_only`, `shuffled_piece_view`). The `random_view_maps` ablation freezes the cross-view maps via a seeded RNG (`model.random_map_seed`); changing that seed lets you re-run the falsification check with a different fixed map population.
