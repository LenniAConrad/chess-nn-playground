# Implementation Notes

- Bespoke model code: `src/chess_nn_playground/models/centered_tempo_odd_interventional_bottleneck.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i041_centered_tempo_odd_interventional_bottleneck/model.py`.
- Registered model name: `centered_tempo_odd_interventional_bottleneck`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0729_tuesday_pacific_tempo_odd_bottleneck.md`.
- Encoding contract: simple_18 only. The deterministic adapter fails
  closed on any other channel layout, so LC0 layouts are rejected
  rather than silently toggling unsupported planes.
- Inputs consumed by the model: board tensor only. CRTK metadata,
  source labels, engine evaluations, mate flags, legal-move counts,
  and verification metadata are never used as model input.
- Compute: one shared encoder call on the four-view concatenated batch
  of shape `(4B, 18, 8, 8)`. An optional `encoder_chunk_size` config
  field can break the encoder pass into smaller chunks while preserving
  the deterministic pairing order `[x, tau(x), nu(x), tau(nu(x))]`.
