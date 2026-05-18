# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/halfka_dual_stream_lc0.py`.
- Idea-local wrapper: `ideas/registry/i243_halfka_dual_stream_lc0/model.py`.
- Registry key: `halfka_dual_stream_lc0`.
- Reuses `DualStreamFeatureBuilder` and `StreamEncoder` from `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py` (i193) for the deterministic exchange / king geometry planes and the conv stream encoders.
- Board-only: the architecture does not consume engine, verification, source, or CRTK metadata as input.
- HalfKA front-end: two learnable embedding tables of shape `(64, 64, 6, embed_dim)` per side, indexed by `(king_square, piece_square, piece_type)`. King squares are extracted as argmax over the king planes; that yields the same per-side accumulator that a true HalfKA implementation would compute when the board is reachable from a legal position.
- The accumulator is split per square (each `piece_square` contributes to exactly the embedding at that square) so the output reshapes to `(B, 2 * embed_dim, 8, 8)` for conv consumption. This is the same per-square reconstruction the architecture math thesis describes.
- The compact CPU-testable variant uses `embed_dim=16`, `backbone_channels=48`, `backbone_depth=2`, `head_hidden=96`, `policy_dim=32` (~0.9M params). The engine-grade scaled variant in the architecture notes would use `embed_dim=256`, `backbone_channels=128`, `backbone_depth=6`, `policy_dim=1858`; the puzzle_binary implementation only builds the compact variant.
- Ablation modes (`ablation`): `none` (default), `no_halfka`, `no_dual_stream`, `no_residual`, `puzzle_only`.
- Config keys: `embed_dim` (alias `channels`), `backbone_channels`, `backbone_depth` (alias `depth`), `head_hidden` (alias `hidden_dim`), `dropout`, `policy_dim`, `use_batchnorm`, `ablation`.
- LC0 heads (`value_wdl_logits`, `policy_logits`) are exposed as diagnostic outputs; the puzzle_binary trainer does not consume them. Engine-grade training that does is out of scope for this implementation.
