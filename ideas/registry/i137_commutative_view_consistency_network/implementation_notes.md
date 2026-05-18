# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/commutative_view_consistency.py`.
- Registry key: `commutative_view_consistency_network`.
- Builder: `build_commutative_view_consistency_network_from_config`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.
- Batch candidate: `Commutative View-Consistency Network`.
- Input contract: simple_18 only (18 planes, 8x8); the model raises on other encodings or input-channel counts.
- Output contract: puzzle_binary one-logit (`num_classes = 1`); the model raises if requested otherwise. The forward dict includes the puzzle logit, probability, stacked per-view latents, per-view RMS norms, per-defect statistics tensor (`defect_stats` with shape `(B, 9, 5)`), per-defect convenience views (`defect_l2`, `defect_l1`, `defect_cosine`), aggregate `consistency_energy`, `mean_defect_l1`, `mean_defect_cosine`, bookkeeping diagnostics `commutative_view_ablation`, `commutative_view_count`, `mechanism_energy`, `proposal_profile_strength`, `proposal_keyword_count`, the raw `z_<view>` latents, and the raw `defect_<i>` residual vectors.
- Trainable parameters: the five view encoders, the eight low-rank `A_{u → v}` maps, and the MLP head. The line/region/count deterministic summary tables and the coordinate buffer of the piece DeepSets are non-learnable (registered as buffers or computed inline).
- View construction: deterministic, computed at module init / forward. The piece DeepSets encoder builds per-square tokens from the 12 piece planes + 4 coordinate features (rank, file, centre distance, square parity) and mean-aggregates them under an occupancy mask so the operator is permutation-invariant in the piece token set.
- Cross-view maps: each `_LowRankMap` is `Linear(D, rank, bias=False)` followed by `Linear(rank, D, bias=True)`. In the `random_view_maps` ablation, every map is frozen to scale-matched random values from a seeded `torch.Generator` (`random_map_seed`).
- This idea is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Ablation map (see `CommutativeViewConsistencyNetwork.ABLATIONS`): `none`, `views_only_no_defects`, `single_square_view`, `random_view_maps`, `count_to_all_only`, `shuffled_piece_view`. The `views_only_no_defects` and `single_square_view` ablations are the load-bearing comparisons; `random_view_maps` is the falsification check for the learned cross-view maps; `count_to_all_only` and `shuffled_piece_view` are material-shortcut and piece-geometry checks.
- Tests live at `tests/test_commutative_view_consistency_network.py` and cover the registry contract, configuration parsing, forward keys, gradient flow through the encoders / maps / head, the buffer-vs-parameter discipline of the deterministic summary tables, every ablation's documented behavior, and the idea-folder conformance audits.
