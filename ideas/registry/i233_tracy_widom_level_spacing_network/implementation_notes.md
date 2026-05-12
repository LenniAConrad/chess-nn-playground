# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/trunk/tracy_widom_level_spacing_network.py`.
- Builder: `build_tracy_widom_level_spacing_network_from_config`.
- Registry key: `tracy_widom_level_spacing_network`.
- Idea-local wrapper: `ideas/registry/i233_tracy_widom_level_spacing_network/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-05-05_1610_tuesday_local_tracy_widom_level_spacing.md`.
- Input is the board tensor only; CRTK / source / engine / verification
  metadata is reporting-only and never consumed as model input.
- The Hermitian operator is obtained from two convolutional `1 x 1`
  projections of the trunk feature map (left / right embeddings of size
  `embedding_dim`), giving an asymmetric `64 x 64` matrix that is then
  symmetrised. The model uses `torch.linalg.eigvalsh` on the resulting
  Hermitian matrix to compute the spectrum.
- Unfolding is a 5-point boxcar smoothing of the empirical staircase
  (`unfolding_window`); the spacing scale is normalised to mean 1.
- Output keys (in addition to `logits`):
  - `tracy_widom_mean_spacing_ratio` (B,)
  - `tracy_widom_spacing_histogram` (B, `spacing_histogram_bins`)
  - `tracy_widom_spectral_form_factor` (B, `num_form_factor_taps`)
  - `tracy_widom_regime_softmax` (B, 3) for `[Poisson, GOE, GUE]`
  - `tracy_widom_poisson_loglik`, `tracy_widom_goe_loglik`,
    `tracy_widom_gue_loglik` (B,)
