# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/chess_geometry_transport.py`.
- Idea-local wrapper:
  `ideas/all_ideas/registry/i034_entropic_chess_geometry_transport_network/model.py`
  (exports `build_model_from_config(config)`).
- Registered model name: `entropic_chess_geometry_transport_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0703_tuesday_los_angeles_geom_ot.md`.
- The model is intentionally board-only and never consumes engine,
  verification, source, or CRTK metadata as input.
- Only the `simple_18` 18-channel encoding is supported; constructing
  `EncodingSemanticAdapter` with another encoding or channel count
  raises `ValueError` so unknown deterministic channel semantics fail
  closed.
- Atom budgets are deterministic: `max_sources=16`,
  `max_targets=40`, with target-role layout
  `(king_square, king_ring(8), top heavy/minor/pawn material(23),
  promotion_anchor(8))`. The atom builder keeps a sentinel king atom
  at `_CENTER_SQUARE=27` if no side-to-move piece is found, so
  marginals never collapse to zero.
- Distance tables (`_piece_distance_tables`) are precomputed buffers:
  knight BFS, rook line distance, bishop colour-aware distance, queen
  `min(rook, bishop)`, king Chebyshev, and directional pawn distance
  with a `distance_cap` (default 8.0). A Manhattan correction table is
  also registered as a buffer.
- Sinkhorn is unrolled in log domain. The kernel uses
  `(-cost / epsilon).masked_fill(~valid, -1e9)` so masked pairs do
  not pollute the iterations. Final plan entries are renormalized to
  the simplex over `(S, T)`.
- `cost_ablation_mode` accepts `none`, `uniform`, and
  `random_cost_histogram_preserving`. Default is `none`; the
  randomization permutes valid pairs by `(piece_type, target_role)`
  bin with a deterministic shift indexed by batch position.
- Outputs include `logits` plus diagnostics `transport_cost`,
  `transport_entropy`, `transport_source_concentration`,
  `transport_target_concentration`, `transport_king_flow`, and
  `transport_role_pressure`. With `num_classes == 1` the logit is
  squeezed to shape `(batch,)` for the puzzle-binary trainer.
