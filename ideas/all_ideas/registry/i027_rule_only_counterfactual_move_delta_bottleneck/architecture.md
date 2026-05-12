# Architecture

`Rule-Only Counterfactual Move-Delta Bottleneck` (CDBN) classifies puzzle-likeness from a current-board `simple_18` tensor only. The architecture follows the markdown thesis: enumerate the side-to-move pseudo-legal one-ply move set, encode each move's local delta in board context, and aggregate the multiset through a *sparse* permutation-invariant move-cone bottleneck. Engine analysis, source labels, self-check filtering and check/mate/stalemate oracles are never used as inputs.

## Forward Pass

1. **Board adapter.** A `simple_18` parser builds the current board state `B(x) = (occupancy_12, side_to_move, castling, en_passant)`. Unsupported encodings or channel orders fail closed with `ValueError`.
2. **Pseudo-legal move-delta enumerator.** For the side to move, generate pseudo-legal one-ply moves with rule-only piece movement: pawn pushes/captures (with promotions), leaper moves for knights and king, slider rays stopped by the first occupied square, and optional castling candidates from the castling planes only. No engine evaluation, self-check filtering, or terminal-state oracles. Moves are emitted in deterministic `(piece, from, special, to, promotion)` order with a padded validity mask `valid_mask`.
3. **Board context encoder.** A small residual CNN with coordinate planes produces an `8x8xH` square map `F` and a parent-board context vector `z` of dimension `D`.
4. **Move-delta tuple encoder.** For every move `m` we gather the from/to square features `F_from`, `F_to`, the finite difference `F_to - F_from` and a slider-path mean of `F` along the (exclusive) ray between source and destination. We also embed the deterministic descriptors `(piece, capture, promotion, special, relative bucket, normalised delta-rank/file, capture/promotion indicators)` and the broadcast global vector `z`. An MLP returns per-move response vectors `r in R^R`.
5. **Move-cone bottleneck.** Padded tokens are masked. The pool computes:
   - per-move scores `s_m = score_mlp([r_m, z])`,
   - a sparse weighting `alpha_m = masked_sparsemax(s_m / temperature)` (entmax-1.5 is also supported),
   - the sparse bottleneck vector `b_sparse = sum_m alpha_m r_m`,
   - the masked mean `b_mean = mean_m r_m`,
   - the masked second moment `b_second = mean_m r_m^2`,
   - the anisotropy scalar `kappa = max_m s_m - logmeanexp_m s_m`.
6. **Classifier head.** An MLP receives `[z, b_sparse, b_mean, b_second, kappa]` and returns the puzzle logit(s).

The forward returns a dict with `logits` shaped `(B,)` (for `num_classes == 1`) plus diagnostic tensors:

- `move_cone_kappa`, `move_cone_score_max`, `move_cone_score_logmeanexp`,
- `move_cone_sparse_active_count`, `move_cone_alpha_max`, `move_cone_alpha_entropy`,
- `move_cone_b_sparse_norm`, `move_cone_b_mean_norm`, `move_cone_b_second_sum`,
- `pseudo_legal_move_count`, `capture_move_fraction`, `promotion_move_fraction`.

## Why this is bespoke

The central operator is a deterministic chess intervention (`T_m x - x`) plus a *sparse* permutation-invariant pool over the move-delta multiset. The model is not a CNN over the static board, not a square ViT, not an attack-defense sheaf, and is materially distinct from `one_ply_counterfactual_move_landscape_network` (i025, free-energy attention pool) and `counterfactual_move_delta_spectrum_network` (i026, covariance/eigen-spectrum pool). The sparsemax/entmax bottleneck and the `kappa` anisotropy scalar are the falsifiable mechanism that the markdown thesis tests with its degree-preserving move-delta shuffle ablation.

## Implementation Binding

- Registered model name: `rule_only_counterfactual_move_delta_bottleneck`.
- Source implementation file: `src/chess_nn_playground/models/counterfactual_delta_bottleneck.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i027_rule_only_counterfactual_move_delta_bottleneck/model.py` — exposes `build_model_from_config(config)` that delegates to `build_counterfactual_delta_bottleneck_from_config`.
- Reused primitives: the `Simple18BoardAdapter` and `PseudoLegalDeltaEnumerator` from `src/chess_nn_playground/models/move_landscape_net.py` provide the rule-only current-board parser and the deterministic pseudo-legal move enumerator. The board context encoder, slider-path mean, move-delta tuple encoder, masked sparsemax/entmax-1.5, move-cone bottleneck and classifier head are bespoke to this idea.
