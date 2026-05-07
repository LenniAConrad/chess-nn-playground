# Architecture

`Counterfactual Move-Delta Spectrum Network` operates on the current
`simple_18` board tensor only. It enumerates the rule-only one-ply move-delta
neighbourhood of the side to move, encodes a learned response vector for every
candidate delta, and pools the move-set with a covariance/eigen-spectrum
operator. Engine analysis, source labels and self-check filtering are never
used as inputs.

## Forward Pass

1. **Board adapter.** A `simple_18` parser builds the current board state
   `B(x) = (occupancy_12, side_to_move, castling, en_passant)`. Unsupported
   encodings or channel orders fail closed with `ValueError`.
2. **Pseudo-legal move-delta enumerator.** For the side to move, generate
   pseudo-legal one-ply move deltas using rule-only piece movement: pawn
   pushes/captures (with promotions), leaper moves for knights and king,
   slider rays stopped by the first occupied square, and optional castling
   candidates from the castling planes. No engine evaluation, self-check
   filtering, or checkmate/stalemate oracles. Moves are emitted in
   deterministic `(piece, from, special, to, promotion)` order with a padded
   validity mask.
3. **Board stem.** A small convolutional stem produces an `8x8xd_sq` square
   feature map `H_sq` and a global feature `g` of dimension `d_g`. Coordinate
   planes are concatenated and projected to retain orientation cues.
4. **Move-token encoder.** For every move ``a`` we gather the from/to square
   features `H_from`, `H_to`, the finite difference `H_to - H_from`, the
   broadcast global vector `g`, and deterministic move descriptors
   `eta(x, a)` (piece id, captured piece, promotion id, special-move id,
   relative-square bucket, normalised delta-rank/file, capture and promotion
   indicators). An MLP produces per-move response vectors `r in R^k`.
5. **Counterfactual spectrum pool.** Padded tokens are masked. The pool
   computes
   - masked mean `r_mean`,
   - masked variance `r_var` (per-coordinate),
   - masked max `r_max` (with safe fallback when no valid token exists),
   - the uniform-weighted covariance
     `K = sum_a w_a (r_a - r_mean)(r_a - r_mean)^T + eps I`
     and its eigenvalues via `torch.linalg.eigvalsh` (sorted descending),
   - spectral statistics: `trace(K)`, `lambda_1 / trace(K)`, the participation
     ratio `(trace K)^2 / trace(K^2)`, normalised spectral entropy
     `-sum_i tilde_lambda_i log tilde_lambda_i`, and the Frobenius norm.
6. **Spectrum head.** A LayerNorm + 2-layer MLP receives `g`, `r_mean`,
   `r_max`, `r_var`, the eigenvalues, and the spectral statistic vector and
   emits one logit (puzzle-binary) or `num_classes` logits if requested.

The model returns a dictionary containing `logits` shaped `(B,)` and
diagnostic scalars per board: `spectrum_trace`, `spectrum_leading_fraction`,
`spectrum_participation_ratio`, `spectrum_entropy`, `spectrum_frobenius_norm`,
`spectrum_top_eigenvalue`, `spectrum_response_mean_norm`,
`spectrum_response_max_norm`, `spectrum_response_var_sum`,
`pseudo_legal_move_count`, `capture_move_fraction`, `promotion_move_fraction`,
`trace_penalty_beta`, `mechanism_energy`, `proposal_profile_strength`, and
`proposal_keyword_count`.

## Why this is not a generic baseline

The CNN stem is small and only produces square embeddings and a global
feature; the central operator is the rule-only one-ply move-delta covariance
spectrum, not deeper convolutions, a square ViT, an attack-defense sheaf, or
a DeepSets/free-energy pool. This separates the architecture from
`one_ply_counterfactual_move_landscape_network` (i025), which uses an
energy-based attention pool with free-energy and top-2 gap statistics rather
than a covariance/eigen-spectrum.

## Implementation Binding

- Registered model name: `counterfactual_move_delta_spectrum_network`.
- Source implementation file: `src/chess_nn_playground/models/counterfactual_move_delta_spectrum.py`.
- Idea-local wrapper: `ideas/i026_counterfactual_move_delta_spectrum_network/model.py`
  — exposes `build_model_from_config(config)` that delegates to
  `build_counterfactual_move_delta_spectrum_network_from_config`.
- Reused primitives: the `Simple18BoardAdapter` and
  `PseudoLegalDeltaEnumerator` from
  `src/chess_nn_playground/models/move_landscape_net.py` provide the rule-only
  current-board parser and the deterministic pseudo-legal delta enumerator.
  The covariance/eigen-spectrum pool, move-response encoder, and classifier
  head are bespoke to this idea.
