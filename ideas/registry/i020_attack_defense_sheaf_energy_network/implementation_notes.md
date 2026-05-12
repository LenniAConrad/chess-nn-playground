# Implementation Notes

- Central code: `src/chess_nn_playground/models/attack_defense_sheaf.py`.
- Idea-local wrapper: `ideas/registry/i020_attack_defense_sheaf_energy_network/model.py`.
- Registry key: `attack_defense_sheaf_energy_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0255_tuesday_local_attack_defense_sheaf.md`.
- The model is intentionally board-only: it does not consume engine,
  verification, source, or CRTK metadata as input.
- Static chess geometry (rays, knight, king, pawn-attack offsets) and
  per-square coordinate features are precomputed once at construction and
  registered as non-persistent buffers; nothing depends on per-sample
  metadata.
- The visibility gate uses a learned occupancy proxy
  `o_v = sigmoid(w_occ^T h_v)` and a multiplicative blocker product, with
  padded blocker indices replaced by `1.0` so they cannot leak into `q_e`.
