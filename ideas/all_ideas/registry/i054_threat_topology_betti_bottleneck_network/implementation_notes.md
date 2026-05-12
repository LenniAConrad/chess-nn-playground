# Implementation Notes

- Central code: `src/chess_nn_playground/models/threat_topology_betti.py`.
- Registry key: `threat_topology_betti_bottleneck_network`.
- Idea wrapper: `ideas/all_ideas/registry/i054_threat_topology_betti_bottleneck_network/model.py`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0814_tuesday_los_angeles_threat_topology.md`.

The deterministic branch currently supports `simple_18` only. It decodes channels `0..5` as white pieces, `6..11` as black pieces, and channel `12` as side-to-move. Castling and en-passant planes are available to the CNN stem but are not interpreted by the rule topology branch. LC0-style encodings fail closed unless a future adapter provides explicit current-piece channel semantics.

Pseudo-legal pressure is generated directly from current occupancy. Sliding attacks include the blocker square and stop beyond it. This is intentionally not legal move generation and does not inspect check, mate, engine, verification, source, or label metadata.

`RankCubicalBettiEncoder` uses deterministic square-index tie breaking for top-k masks. `beta0` is computed by bounded 4-neighbor label propagation on the 8x8 active-cell mask. `beta1` is computed from the cubical Euler relation `beta1 = beta0 - V + E - C`, with unique cubical vertices and edges counted from active cells.

The model returns a dictionary because the project research models expose diagnostics. `output["logits"]` follows the repo's one-logit puzzle-binary BCE contract; `output["two_class_logits"]` preserves the packet's two-class internal classifier view.
