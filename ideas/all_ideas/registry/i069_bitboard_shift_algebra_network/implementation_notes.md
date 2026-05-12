# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/bitboard_shift_algebra.py`.
- Registry key: `bitboard_shift_algebra_network`.
- Idea-local wrapper: `ideas/all_ideas/registry/i069_bitboard_shift_algebra_network/model.py` (thin `build_model_from_config`).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2131_friday_shanghai_bitboard_shift_algebra.md`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The 16 single-step shift operators (8 king/orthogonal/diagonal + 8 knight L-shapes) are constructed geometrically as 64-square gather indices with masked wraparound; they carry no learned parameters.
- Path basis is fixed at 12 families and depth is capped at `path_depth_max <= 3`, keeping the polynomial degree small.
- Coefficient emitter consumes the pooled stem (`mean`, `max`) and a coarse material summary; coefficients are normalized by `tanh / sqrt(P)` by default (or `softmax` over `P`).
- Side-relative pawn-capture paths are realized by mixing white (`nw`/`ne`) and black (`se`/`sw`) shifts under the side-to-move plane so the model does not need to learn pawn direction separately.
- Ablations (`cnn_only`, `random_shift_bank`, `orthogonal_only`, `one_step_only`, `fixed_alpha`, `no_gate`, `dense_conv_matched`) are exposed by the bespoke builder for the packet's central falsifiers.
