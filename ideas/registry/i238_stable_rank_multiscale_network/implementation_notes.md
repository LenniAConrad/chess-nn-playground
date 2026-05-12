# Implementation Notes

- **Bespoke module**: `src/chess_nn_playground/models/stable_rank_multiscale.py`.
- **Class**: `StableRankMultiscaleNetwork`.
- **Builder**: `build_stable_rank_multiscale_from_config` (registered in `registry.py`).
- **Registry key**: `stable_rank_multiscale_network`.
- **Source packet**: `ideas/research/packets/classic/chess_nn_research_2026-05-05_1710_tuesday_local_stable_rank_multiscale.md`.
- This is a real implementation of the algebraic operator from the source packet
  -- not a generic ResearchPacketProbe profile.
- Forward and backward pass have been smoke-tested with a 4-sample batch and BCE
  loss; gradients flow through the bespoke linear-algebra primitives correctly.
- Inputs: only the current-board encoding tensor; no engine, no CRTK metadata, no
  source labels.
