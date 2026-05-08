# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/zobrist_kernel_feature_network.py`.
- Idea-local wrapper: `ideas/i135_zobrist_kernel_feature_network/model.py`.
- Registry key: `zobrist_kernel_feature_network` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.
- Batch candidate: `Zobrist Kernel Feature Network` (rank 5).
- The model is intentionally board-only and consumes only the 12 piece planes
  of the `simple_18` tensor; engine, verification, source, and CRTK metadata
  are never used as model input.
- Only the classifier MLP is trainable. The `M` Zobrist code banks
  (`zobrist_codes`), the per-bank random projections (`projection`), and the
  per-bank phase biases (`phase_bias`) are sampled once with a fixed seed in
  `__init__` and registered as buffers (`persistent=True`).
