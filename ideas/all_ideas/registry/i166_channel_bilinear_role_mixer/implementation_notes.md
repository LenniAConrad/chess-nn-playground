# Implementation Notes

- Central code: `src/chess_nn_playground/models/channel_bilinear_role_mixer.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i166_channel_bilinear_role_mixer/model.py`.
- Registry key: `channel_bilinear_role_mixer`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.
- Batch candidate: `Channel-Bilinear Role Mixer`.
- Board-only: the model consumes the simple_18 current-board tensor and does
  not read engine, verification, source, or CRTK metadata as input.
- The role spatial gates are parameters of shape `(K, 64)` and are softmaxed
  per role at every forward pass, so each gate is a proper distribution over
  the 64 squares.
- The bilinear interaction matrix `M \in R^{K \times K}` is computed by an
  einsum over the rank-`R` views (`P, Q`) of the role summaries; no
  materialised square-pair tensor is constructed.
- Diagnostics returned with the logit (`role_summaries`,
  `bilinear_interaction_matrix`, `bilinear_diag`, `bilinear_energy`,
  `bilinear_off_diag_energy`, `bilinear_asymmetry`, `role_gates`,
  `role_magnitude`, `role_gate_entropy`, `depth_levels`) are intended for
  prediction artifacts and run analysis; they are not used to compute the
  training loss.
