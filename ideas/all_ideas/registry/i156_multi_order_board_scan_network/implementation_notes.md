# Implementation Notes

- Central code: `src/chess_nn_playground/models/multi_order_board_scan_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i156_multi_order_board_scan_network/model.py`.
- Registry key: `multi_order_board_scan_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.
- Batch candidate: `Multi-Order Board Scan Network`.
- The model is board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Five scan orders are used: `rank_major`, `file_major`, `diagonal`, `spiral_from_king`, and `center_out`. Four are static buffer permutations; `spiral_from_king` is per-sample via a precomputed `(64, 64)` lookup table indexed by the side-to-move king square.
