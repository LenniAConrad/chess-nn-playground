# Implementation Notes

- Central code: `src/chess_nn_playground/models/causal_piece_derivative_network.py`.
- Idea-local wrapper: `ideas/registry/i179_causal_piece_derivative_network/model.py`.
- Registry key: `causal_piece_derivative_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.
- Batch candidate: `Causal Piece-Derivative Network`.
- This is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
- Intervention types are implemented as a learnable embedding fed
  into a *shared* delta encoder; the trunk is computed once per
  batch and never re-run per intervention, matching the packet's
  cost-control requirement.
- Required ablations: `none`, `random_candidates`, `no_delta_readout`,
  `full_remove_only`, `candidate_k_4`.
