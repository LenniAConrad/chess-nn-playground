# Architecture

`Invertible Board Coupling Network` materialises the reversible-encoder
thesis from `math_thesis.md`. Standard encoders can discard information
early, which makes it hard to know whether a model latched onto legitimate
current-board structure or fragile shortcuts. This network preserves
information by construction with a stack of invertible affine coupling
blocks and invertible 1x1 channel mixings, then classifies from latent
statistics together with explicit per-block coupling-scale diagnostics.

- Input: simple_18 board tensor `(B, 18, 8, 8)`. CRTK/source metadata is
  reporting-only and never used as input.
- Reversible projection: simple_18 is zero-padded to width `D` (default
  `D = 64`). Padding is information-preserving; the inverse keeps the first
  18 channels.
- Invertible 1x1 channel mixing: each `InvertibleConv1x1` is parameterised
  by a learned full square matrix initialised to a random orthogonal matrix
  and inverted with `torch.linalg.inv` for reconstruction.
- Affine coupling block:
  ```text
  y_a = x_a
  y_b = x_b * exp(s(x_a)) + t(x_a)
  ```
  with `s = scale_clamp * tanh(raw_s)` so the affine transform is bounded
  (the implementation-notes recipe from the source packet). The active half
  alternates across blocks via the `swap` flag, which is the channel-split
  alternation called for in the architecture sketch. The `s,t` predictor
  is a 3-conv stack whose final layer is zero-initialised, so each block
  is the identity at init and deep stacks are stable from the first step.
- Frozen-inverse check: after the forward pass, the encoder is fully
  inverted (`final_mixing⁻¹` followed by `coupling⁻¹` and `mixing⁻¹` per
  block in reverse) and the per-sample
  `inverse_reconstruction_error = mean |z₀ − recon(latent)|` is reported
  as a diagnostic. At init it sits near machine tolerance, matching the
  packet's `frozen_inverse_check` ablation.
- Latent pool + diagnostics: the final latent is mean+max pooled over the
  board. Per-block `mean_abs_s` and `max_abs_s`, plus aggregate
  `latent_energy`, `latent_peak`, and `inverse_reconstruction_error`,
  form the diagnostic vector.
- Classifier: an MLP head consumes the latent pool and the diagnostic
  vector to return one puzzle logit. A parallel diagnostic-only linear
  branch returns a logit from the diagnostics alone, mirroring the
  `no_scale_stats` ablation framing in the source packet.

## Implementation Binding

- Registered model name: `invertible_board_coupling_network`.
- Source implementation file:
  `src/chess_nn_playground/models/invertible_board_coupling_network.py`.
- Idea-local wrapper: `ideas/registry/i122_invertible_board_coupling_network/model.py`
  (a thin `build_model_from_config(config)` wrapper around
  `build_invertible_board_coupling_network_from_config`).
- The model is registered in `src/chess_nn_playground/models/registry.py`
  and is excluded from `RESEARCH_PACKET_MODEL_NAMES` in
  `src/chess_nn_playground/models/research_packet_registry.py` so the
  implementation-kind audit detects this folder as `bespoke_model`.
- Output contract: returns `{"logits": (B,), ...}` with diagnostic
  tensors `diagnostic_branch_logit`, `latent_energy`, `latent_peak`,
  `inverse_reconstruction_error`, `mean_abs_coupling_scale`,
  `max_abs_coupling_scale` (all shape `(B,)`).
