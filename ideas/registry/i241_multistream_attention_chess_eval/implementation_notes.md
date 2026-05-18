# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/multistream_attention_chess_eval.py`.
- Idea-local wrapper: `ideas/registry/i241_multistream_attention_chess_eval/model.py`.
- Registry key: `multistream_attention_chess_eval`.
- Reuses `DualStreamFeatureBuilder` from `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py` (i193) for deterministic king-conditioned and exchange feature planes and for the attention-bias attack tables.
- Board-only: the architecture does not consume engine, verification, source, or CRTK metadata as input.
- The compact CPU-testable variant uses `embed_dim=64`, `num_heads=4`, two blocks per stream (~270k params). The scaled engine variant in the architecture notes would use `embed_dim=128, blocks=8` and add value+policy heads; only the puzzle_binary variant is built here.
- Ablation modes (`ablation`): `none` (default), `no_chess_bias`, `no_phase_router`, `remove_positional_stream`, `remove_king_stream`, `remove_exchange_stream`, `no_aux_heads`.
- Config keys: `embed_dim` (alias `channels`), `num_heads`, `exchange_blocks`, `king_blocks`, `positional_blocks`, `mlp_ratio`, `dropout`, `aux_loss_weight`, `ablation`.
- The positional stream uses a learnable per-head relative rank/file attention bias of shape `(num_heads, 15, 15)` per axis. Tied across heads sharing offsets via direct indexing into the bias tables.
- The aux head output `aux_loss_weight` is a broadcast scalar (configurable) that downstream scaled trainers can use to weight per-stream aux losses; the puzzle_binary trainer ignores it.
