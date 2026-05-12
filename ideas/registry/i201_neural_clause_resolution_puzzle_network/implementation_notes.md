# Implementation Notes

- Central code: `src/chess_nn_playground/models/neural_clause_resolution_puzzle_network.py`.
- Idea-local wrapper: `ideas/registry/i201_neural_clause_resolution_puzzle_network/model.py`.
- Registry key: `neural_clause_resolution_puzzle_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`.
- Batch candidate: `Neural Clause-Resolution Puzzle Network`.
- Strictly board-only: CRTK / source / verification / engine metadata is
  reporting-only and never enters the model.
- The bespoke model defines `NeuralClauseResolutionPuzzleNetwork` with a
  compact convolutional trunk, a typed-predicate embedding table, soft
  head/body clause queries, row-stochastic spatial relation kernels and
  `resolution_rounds` soft Horn updates over the unary and global fact
  bases.
