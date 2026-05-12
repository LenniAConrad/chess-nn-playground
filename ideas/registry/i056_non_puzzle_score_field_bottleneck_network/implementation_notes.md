# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/non_puzzle_score_field_bottleneck.py`.
- Registry key: `non_puzzle_score_field_bottleneck_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0922_tuesday_local_nonpuzzle_score_field.md`.
- Idea-local wrapper: `ideas/registry/i056_non_puzzle_score_field_bottleneck_network/model.py` calls
  `build_non_puzzle_score_field_bottleneck_network_from_config`.
- Input contract: simple_18 boards `(B, 18, 8, 8)` only. `fail_closed_on_unknown_encoding=True`
  raises `ValueError` on any other channel count. CRTK/source/engine metadata is never read.
- Two-stage training helpers: call `model.denoising_score_matching_loss(clean_board, binary_label)`
  during the pretraining stage. Pass the binary label tensor so the class-0-only filter is
  applied. Then `model.freeze_score_prior()` to lock the denoiser before the supervised
  classifier stage.
- The forward pass wraps the denoiser in `torch.no_grad()` automatically when the denoiser
  has no trainable parameters and the model is in training mode, which makes the supervised
  trainer correct out-of-the-box once `freeze_score_prior()` has been called.
- Sigmas default to `[0.05, 0.10, 0.20]`. Override via `model.noise_sigmas` in config.
- The model returns a `dict[str, torch.Tensor]` with `logits` of shape `(B,)` for
  `num_classes=1`, plus diagnostic keys consumed by the shared reporting pipeline.
