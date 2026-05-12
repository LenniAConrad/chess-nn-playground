# Implementation Notes

- Central code: `src/chess_nn_playground/models/tactical_hessian_spectrum_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i199_tactical_hessian_spectrum_network/model.py`.
- Registry key: `tactical_hessian_spectrum_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0109_saturday_shanghai_high_upside_puzzle_batch_4.md`.
- Batch candidate: `Tactical Hessian Spectrum Network`.
- Strictly board-only: the model never consumes engine, verification, source, or CRTK metadata.
- Forward pass evaluates the encoder once on a batched stack of variants
  `{x, x ± eps D_k, x + eps (D_i + D_j) for i < j}`; with the default
  `num_directions = 4` that is `15` board variants per sample. This is
  the principal compute knob, alongside `eps` and the trunk
  `channels` / `depth`.
