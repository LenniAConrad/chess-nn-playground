# Implementation Notes

- Bespoke model: `src/chess_nn_playground/models/trunk/traced_threat_motif.py`.
- Idea-local wrapper: `ideas/registry/i088_traced_threat_motif_network/model.py`.
- Registry key: `traced_threat_motif_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-28_0857_tuesday_new_york_trace_motif.md`.
- The model is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
- Geometry masks (legal moves, sliding-piece between-square clearance,
  pawn double-step middle-square clearance) are precomputed once at
  module construction and reused for every batch.
- The motif vocabulary in `MOTIF_WORDS` is fixed; only the relation
  gate, the group-mixing softmax weights, the material value vector,
  and the head are trainable.
