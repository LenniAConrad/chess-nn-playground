# Implementation Notes

- Central code: `src/chess_nn_playground/models/hypercolumn_square_readout_cnn.py`.
- Registry key: `hypercolumn_square_readout_cnn`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.
- Batch candidate: `Hypercolumn Square Readout CNN`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The model uses a residual CNN trunk with saved intermediate 8x8 feature maps, per-depth 1x1 projections, concatenated square hypercolumns, a square evidence head, and a global MLP over mean, max, and top-k square evidence pools.
