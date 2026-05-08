# Implementation Notes

- Central code: `src/chess_nn_playground/models/exchange_soundness_graph_network.py`.
- Idea-local wrapper: `ideas/i187_exchange_soundness_graph_network/model.py`.
- Registry key: `exchange_soundness_graph_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Exchange-Soundness Graph Network`.
- Bespoke board-only architecture: it consumes only the `simple_18`
  current-board tensor (piece planes + side-to-move). CRTK / engine /
  source / verification metadata are never used as model input.
