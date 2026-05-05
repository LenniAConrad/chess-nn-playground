# Implementation Notes

- **Bespoke module**: `src/chess_nn_playground/models/cayley_hamilton_coeffs.py`.
- **Class**: `CayleyHamiltonCoefficientNetwork`.
- **Builder**: `build_cayley_hamilton_coeffs_from_config` (registered in `registry.py`).
- **Registry key**: `cayley_hamilton_coeffs_network`.
- **Source packet**: `ideas/research_packets/chess_nn_research_2026-05-05_1720_tuesday_local_cayley_hamilton_coeffs.md`.
- This is a real implementation of the algebraic operator from the source packet
  -- not a generic ResearchPacketProbe profile.
- Forward and backward pass have been smoke-tested with a 4-sample batch and BCE
  loss; gradients flow through the bespoke linear-algebra primitives correctly.
- Inputs: only the current-board encoding tensor; no engine, no CRTK metadata, no
  source labels.
