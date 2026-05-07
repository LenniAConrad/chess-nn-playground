# Architecture

`Entropic Piece-Target Transport Bottleneck` implements a board-only entropic optimal-transport classifier that couples canonical piece-source measures to deterministic king/value/promotion target anchors over a fixed bank of empty-board chess distance matrices.

## Implementation Binding

- Registered model name: `entropic_piece_target_transport_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/entropic_piece_target_transport_bottleneck.py`
- Idea-local wrapper: `ideas/i029_entropic_piece_target_transport_bottleneck/model.py`

## Components

- `Simple18Adapter`: extracts the 12 piece planes and side-to-move scalar from the `simple_18` board tensor. CRTK/source/engine metadata is never used as model input. Unknown channel maps fail closed.
- `SideToMoveCanonicalizer`: when black is to move, swaps colors and flips ranks (vertical flip) so the side-to-move pieces always occupy planes `0..5` of the canonical view and our pawns push toward canonical row `0`.
- `BoardStem`: small `Conv3x3 -> GroupNorm -> GELU` stack of `depth` layers with `channels` features. Operates on the raw `simple_18` tensor and provides square features `h in R^{D x 8 x 8}` and a global pool `h_pool in R^{D}`.
- `_build_source_masks`: deterministic occupancy masks for the six canonical groups `us_sliders` (B+R+Q), `us_leapers` (N+K), `us_pawns`, `them_sliders`, `them_leapers`, `them_pawns`.
- `MaskedSourceMeasure`: per-group salience head `Linear(D, NUM_GROUPS)` over flattened stem features. Logits are masked by occupancy and softmaxed; empty groups deterministically fall back to the uniform distribution (no empty-group flag is exposed to the classifier, by design).
- `_build_target_anchors`: deterministic measures `nu_a` for `them_king_zone`, `them_value`, `us_king_zone`, `us_value`, `us_promotion_rank`, `them_promotion_rank`. King zones use `exp(-beta * Chebyshev(square, king_square))`; value anchors use the fixed nominal piece values `Q=9, R=5, B/N=3, P=1` and fall back to the matching king-zone measure when the side has no non-king material.
- `ChessMetricBank`: precomputed `[NUM_METRICS=7, 64, 64]` empty-board distances (`king`, `manhattan`, `rook`, `bishop` with opposite-color cap, BFS `knight`, `pawn_us`, `pawn_them`). Per-group cost mixtures use `softplus(beta_g0 + sum_r softplus(alpha_{g,r}) * D_r)` with nonnegative alphas.
- `LogSinkhorn`: log-domain Sinkhorn solver with `epsilon` and `iters` config knobs. Computes the entropic optimal coupling for all 12 source-target pairs simultaneously over `[B, P, 64, 64]`.
- Pair list `DEFAULT_PAIRS` (length `P=12`):
  ```
  us_sliders -> them_king_zone
  us_leapers -> them_king_zone
  us_pawns   -> them_king_zone
  us_sliders -> them_value
  us_leapers -> them_value
  us_pawns   -> them_value
  them_sliders -> us_king_zone
  them_leapers -> us_king_zone
  them_pawns   -> us_king_zone
  them_sliders -> us_value
  us_pawns     -> us_promotion_rank
  them_pawns   -> them_promotion_rank
  ```
- Per-pair transport summaries `tau` (5 features per pair): `ot_cost`, `prod_cost = <mu otimes nu, C>`, `transport_gap = (prod_cost - ot_cost)_+`, normalized `plan_entropy`, and `sharpness = sum Pi^2`.
- `Classifier`: `Linear(P*5 + D, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, num_classes)` returning one puzzle logit (`num_classes: 1`).

## Forward Contract

```text
output = model(x)
x.shape == (batch, input_channels=18, 8, 8)
output["logits"].shape == (batch,)        # because num_classes == 1
```

The diagnostics dictionary additionally exposes `transport_imbalance`, `sheaf_tension`, `symmetry_residual`, `topology_pressure`, `ray_language_energy`, `information_surprisal`, `sparse_certificate_energy`, `rank_file_imbalance`, `king_ring_pressure`, `reply_pressure`, `defense_gap`, `mechanism_energy`, `proposal_profile_strength`, and `proposal_keyword_count` derived from per-pair transport gaps, costs, entropies, and sharpness so downstream artifact reporting stays compatible with the puzzle_binary trainer.

## Symmetry

Only side-to-move color/rank canonicalization is enforced. Horizontal file mirror, full board rotation, or color-blind symmetries are not used because pawns, castling, and side-to-move break those symmetries. The chess-metric bank is fixed at construction time, so any geometry-destroying ablation (cost permutation) acts only on the registered metric structure, not on the rest of the network.
