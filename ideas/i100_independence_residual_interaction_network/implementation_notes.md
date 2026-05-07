# Implementation Notes

- Central code: `src/chess_nn_playground/models/independence_residual.py`.
- Registry key: `independence_residual_interaction_network`.
- Idea wrapper: `ideas/i100_independence_residual_interaction_network/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.
- Batch candidate: `Independence Residual Interaction Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The implementation builds product-of-marginals expected occupancy maps from
  piece/channel, square, and side-relative rank/file marginals, then classifies
  from signed residual maps and residual interaction statistics.
