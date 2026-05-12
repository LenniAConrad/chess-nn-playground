# Implementation Notes

- Central code: `src/chess_nn_playground/models/adaptive_tactical_resolvent_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i077_adaptive_tactical_resolvent_network/model.py`.
- Registry key: `adaptive_tactical_resolvent_network` (registered as a bespoke builder, not a `ResearchPacketProbe` variant).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_2002_saturday_shanghai_adaptive_tactical_resolvent.md`.
- Board-only by construction: the model consumes `simple_18` board
  tensors and never reads engine, verification, source, or CRTK
  metadata.
- Numerics: 64x64 batched `torch.linalg.solve` for the resolvent. The
  spectral-norm estimator is detached power iteration (cache-friendly
  on GPU) and the dividing scale is clamped at 1.0 so `A_hat` keeps
  spectral radius `<= 1`.
