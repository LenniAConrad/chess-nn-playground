# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/legal_automorphism_quotient_network.py`.
- Registry key: `legal_automorphism_quotient_network`.
- Idea-local wrapper: `ideas/registry/i042_legal_automorphism_quotient_network/model.py` (thin
  delegate around `build_legal_automorphism_quotient_network_from_config`).
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0731_tuesday_los_angeles_orbit_quotient.md`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Channel transforms are registered for `simple_18` only; unsupported encodings fail closed.
