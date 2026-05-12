# Implementation Notes

- Central code: `src/chess_nn_playground/models/spline_board_surface_network.py`.
- Registry key: `spline_board_surface_network`.
- Idea-local wrapper: `ideas/all_ideas/registry/i110_spline_board_surface_network/model.py`
  (a thin `build_model_from_config` over
  `build_spline_board_surface_network_from_config`; no `ResearchPacketProbe`
  is involved).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.
- Batch candidate: `Spline Board Surface Network`.
- Board-only model. CRTK / source / engine / verification metadata is
  reporting-only and is not consumed by the model.
- The Bernstein basis and its pseudoinverse are precomputed once at
  construction time and stored as non-trainable buffers.  Only the residual
  1x1 channel mixer, the residual `LayerNorm` and the head MLP are trained.
