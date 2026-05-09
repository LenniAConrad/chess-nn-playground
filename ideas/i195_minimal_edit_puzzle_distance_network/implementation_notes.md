# Implementation Notes

- Central code: `src/chess_nn_playground/models/minimal_edit_puzzle_distance_network.py`.
- Idea-local wrapper: `ideas/i195_minimal_edit_puzzle_distance_network/model.py`.
- Registry key: `minimal_edit_puzzle_distance_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Minimal-Edit Puzzle Distance Network`.
- Bespoke architecture: a per-square soft edit cost is computed
  between the encoder's per-square symbol distribution `S(x)` and a
  learnable bank of `num_prototypes` puzzle prototypes `P_k`. The
  classifier head reads the soft minimum edit distance
  `D_min(x) = -T * logsumexp(-D_k(x) / T)` and a small set of
  surrounding diagnostics (soft prototype assignment, assignment
  entropy, per-square min-cost map summary, per-prototype distance
  summary statistics).
- Board-only input: CRTK / source / engine metadata is reporting-only.
