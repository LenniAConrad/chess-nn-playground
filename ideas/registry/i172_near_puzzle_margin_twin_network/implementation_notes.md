# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/near_puzzle_margin_twin_network.py`
  (`NearPuzzleMarginTwinNetwork`).
- Idea-local wrapper: `ideas/registry/i172_near_puzzle_margin_twin_network/model.py`
  delegates to `build_near_puzzle_margin_twin_network_from_config`.
- Registry key: `near_puzzle_margin_twin_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.
- Batch candidate: `Near-Puzzle Margin Twin Network` (rank 3).
- Input contract: `simple_18` board tensor `(B, 18, 8, 8)`. CRTK /
  source / engine metadata is reporting-only and never consumed at
  inference.
- Output contract: forward returns a dict with a `(B,)` `logits`
  tensor for the BCE-with-logits puzzle-binary trainer plus the two
  twin latents (`z_ordinary`, `z_tactical`), `puzzle_margin_signal`,
  and per-batch monitoring tensors (`ordinary_norm`, `tactical_norm`,
  `ordinary_tactical_alignment`, `trunk_energy`).
- Margin training: the model itself does not look at group metadata.
  A trainer with reliable `sister_group_id` / `split_group_id` can
  attach pairwise hinge losses on `puzzle_margin_signal` and an
  optional ordinary-latent contrastive term on `z_ordinary`, as
  described in `math_thesis.md`.
