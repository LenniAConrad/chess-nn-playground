# Architecture

`Tracy-Widom Level-Spacing Network` is a bespoke implementation of idea
`i233`. It builds a learned Hermitian operator on the 64 squares of the
board, reads off its spectrum, and classifies puzzle-likeness from
random-matrix-theory bulk invariants of the level-spacing distribution.

## Pipeline

- Input: board tensor `(B, 18, 8, 8)`. CRTK / source metadata is
  reporting-only and never used as model input.
- Convolutional trunk lifts each square to `channels` features.
- Two `1 x 1` convolutions project to "left" and "right" embeddings of
  size `embedding_dim` per square. Their scaled outer product gives an
  asymmetric operator `M` over the 64 squares; symmetrising yields the
  Hermitian operator `H = (M + M^T) / 2` of size `64 x 64`.
- `torch.linalg.eigvalsh(H)` returns the ascending spectrum
  `lambda_1 <= ... <= lambda_64`.
- Unfolding: smooth the empirical staircase with a 5-point boxcar (the
  `unfolding_window` hyperparameter); rescale so unfolded nearest-neighbour
  spacings have mean 1.
- Spacing ratios `r_i = min(s_i, s_{i-1}) / max(s_i, s_{i-1})` give the
  scale-invariant bulk statistic `<r>`.
- A soft RBF histogram over the unfolded spacings (`spacing_histogram_bins`
  bins on `[0, spacing_histogram_max]`) keeps the gradient flowing.
- Spectral form factor
  `K(t_k) = |sum_i exp(2 pi i tilde_lambda_i t_k)|^2 / n`
  is evaluated at `num_form_factor_taps` taps spanning `[t_min, t_max]`.
- Per-spacing log-likelihoods under the Poisson, GOE and GUE surmises are
  summed; their 3-way softmax produces the regime score that distinguishes
  integrable from chaotic positions.
- A LayerNorm + GELU MLP head consumes pooled trunk features (mean and
  max), the spacing histogram, the mean ratio scalar, the form factor
  vector, the per-regime log-likelihoods, and the regime softmax. It
  returns one puzzle logit alongside the bulk-spectrum diagnostics.

## Implementation Binding

- Registered model name: `tracy_widom_level_spacing_network` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/tracy_widom_level_spacing_network.py`
  (`TracyWidomLevelSpacingNetwork` and
  `build_tracy_widom_level_spacing_network_from_config`).
- Idea-local wrapper:
  `ideas/i233_tracy_widom_level_spacing_network/model.py` calls
  `build_tracy_widom_level_spacing_network_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this
  idea.
