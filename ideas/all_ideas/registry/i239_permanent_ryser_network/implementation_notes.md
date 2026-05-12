# Implementation Notes

- **Bespoke module**: `src/chess_nn_playground/models/permanent_ryser.py`.
- **Class**: `PermanentRyserNetwork`.
- **Builder**: `build_permanent_ryser_from_config` (registered in `registry.py`).
- **Registry key**: `permanent_ryser_network`.
- **Source packet**: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-05-05_1715_tuesday_local_permanent_ryser.md`.
- This is a real implementation of the algebraic operator from the source packet
  -- not a generic ResearchPacketProbe profile.
- Forward and backward pass have been smoke-tested with a 4-sample batch and BCE
  loss; gradients flow through the bespoke linear-algebra primitives correctly.
- Inputs: only the current-board encoding tensor; no engine, no CRTK metadata, no
  source labels.
