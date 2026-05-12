# Implementation Notes

- **Bespoke module**: `src/chess_nn_playground/models/hadamard_spectrum.py`.
- **Class**: `HadamardSpectrumNetwork`.
- **Builder**: `build_hadamard_spectrum_from_config` (registered in `registry.py`).
- **Registry key**: `hadamard_spectrum_network`.
- **Source packet**: `ideas/research/packets/classic/chess_nn_research_2026-05-05_1700_tuesday_local_hadamard_walsh_spectrum.md`.
- This is a real implementation of the algebraic operator from the source packet
  -- not a generic ResearchPacketProbe profile.
- Forward and backward pass have been smoke-tested with a 4-sample batch and BCE
  loss; gradients flow through the bespoke linear-algebra primitives correctly.
- Inputs: only the current-board encoding tensor; no engine, no CRTK metadata, no
  source labels.
