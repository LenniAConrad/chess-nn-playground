# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/magnus_bch_coupling_series_network.py`.
- Idea-local wrapper: `ideas/i230_magnus_bch_coupling_series_network/model.py`
  delegates to `build_magnus_bch_coupling_series_network_from_config`.
- Registered model name: `magnus_bch_coupling_series_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1545_tuesday_local_magnus_bch_coupling_series.md`.
- Input is the current-board `simple_18` tensor only; CRTK / source / engine
  metadata is reporting-only and never enters the model.
- The two operators `A`, `B` are produced by linear heads on the pooled
  trunk summary and divided by `1 / sqrt(r)` before spectral clipping
  (default rank `r = 12`). Spectral clipping uses
  `torch.linalg.svdvals(M)[..., 0]` divided by `spectral_clip_per_op`,
  capped at `1`, so each operator satisfies
  `||M||_2 <= spectral_clip_per_op` (default `0.5`). This keeps
  `||A||_2 + ||B||_2 < log 2`, comfortably inside the BCH convergence
  radius.
- Commutators are computed left-to-right exactly as written in the
  source packet so that `c_3a = [A, c_2]`, `c_3b = [B, c_2]`,
  `c_4b = [B, c_3a]`, etc. The BCH log is the explicit weight-4
  truncation `A + B + 1/2 c_2 + (1/12)(c_3a - c_3b) + (1/24) c_4b`.
- Decay ratios use `clamp_min(1e-8)` on the denominator to keep the
  forward pass finite even when one of the commutators happens to be
  numerically zero on a given board.
- Structurally-normalized norms use the multiplicity counts
  `(a, b) = (count of A, count of B)` in each Hall monomial:
  `c_3a = (2, 1)`, `c_3b = (1, 2)`, `c_4a = (3, 1)`, `c_4b = (2, 2)`,
  `c_4c = (2, 2)`, `c_4d = (1, 3)`. This is the source packet's
  recommended "divide by `||A||^k ||B||^{4-k}`" trick for separating
  pure structure from raw operator scale.
