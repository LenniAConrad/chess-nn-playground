# Implementation Notes

- Bespoke model code: `src/chess_nn_playground/models/trunk/piece_liability_gradient_network.py`.
- Idea-local wrapper: `ideas/registry/i202_piece_liability_gradient_network/model.py`.
- Registry key: `piece_liability_gradient_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`.
- Batch candidate: `Piece Liability Gradient Network`.
- The architecture is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The piece-presence mask is built from input planes `0..num_piece_planes-1` of the simple_18 contract (the per-piece bitboards `P, N, B, R, Q, K, p, n, b, r, q, k`).
- The shared `ResearchPacketProbe` wrapper has been removed; this idea now uses a bespoke action-affordance plus liability-propagation pipeline.
