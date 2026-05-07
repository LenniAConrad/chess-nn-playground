# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/entropic_piece_target_transport_bottleneck.py`.
- Idea-local wrapper: `ideas/i029_entropic_piece_target_transport_bottleneck/model.py` (thin `build_model_from_config(config)` that delegates to the bespoke builder).
- Registry key: `entropic_piece_target_transport_bottleneck` (registered in `src/chess_nn_playground/models/registry.py` and excluded from `RESEARCH_PACKET_MODEL_NAMES`).
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0507_tuesday_los_angeles_transport_bottleneck.md`.
- This is intentionally board-only (`simple_18`) and does not consume engine, verification, source, or CRTK metadata as input.
- Sinkhorn runs in log-domain with default `epsilon=0.07` and `iters=8` over all 12 source-target pairs at once. The cost-mixture parameters use `softplus` to keep costs nonnegative.
- The classifier returns `{"logits": [B], ...}` with extra transport diagnostics so the puzzle_binary trainer's prediction-artifact contract continues to work. Empty source groups deterministically fall back to the uniform measure; empty value anchors fall back to the matching king zone.
