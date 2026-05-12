# Implementation Notes

- Bespoke model: `src/chess_nn_playground/models/tempo_alignment_gate_network.py` (`TempoAlignmentGateNetwork`).
- Idea-local wrapper: `ideas/registry/i183_tempo_alignment_gate_network/model.py`.
- Registry key: `tempo_alignment_gate_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.
- Batch candidate: `Tempo-Alignment Gate Network`.
- Board-only: consumes `simple_18` and reads side-to-move from plane 12; CRTK / engine / verification metadata is reporting-only and never enters the model.
