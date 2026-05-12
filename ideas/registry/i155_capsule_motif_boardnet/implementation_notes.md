# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/capsule_motif_boardnet.py`.
- Idea-local wrapper: `ideas/registry/i155_capsule_motif_boardnet/model.py`.
- Registry key: `capsule_motif_boardnet`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.
- Batch candidate: `Capsule Motif BoardNet`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Capsule defaults follow the source packet: `num_primary_caps = 8`, `primary_capsule_dim = 8`, `num_motif_caps = 16`, `motif_capsule_dim = 16`, `routing_iterations = 3`. Override via the matching config keys.
- The squash non-linearity is the standard Sabour-et-al. dynamic-routing form, applied along the capsule-vector axis.
- During routing, only the final iteration uses the live `v_m`; intermediate updates use a detached copy so backprop runs through one round of routing rather than `T` rounds (the standard dynamic-routing recipe).
