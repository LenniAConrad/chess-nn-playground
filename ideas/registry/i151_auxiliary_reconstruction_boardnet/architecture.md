# Architecture

`Auxiliary Reconstruction BoardNet` realises the source packet's
reconstruction-regularised classifier as a bespoke PyTorch model for the
repo's `puzzle_binary` task. A shared convolutional encoder feeds two
heads: a pooled puzzle classifier and a lightweight decoder that
reconstructs a configurable subset of the current-board input planes.
Reconstruction is exposed as an auxiliary training loss so the trunk is
regularised against discarding board detail without ever leaking future
or engine information.

## Implementation Binding

- Registered model name: `auxiliary_reconstruction_boardnet`
- Source implementation file: `src/chess_nn_playground/models/auxiliary_reconstruction_boardnet.py`
- Idea-local wrapper: `ideas/registry/i151_auxiliary_reconstruction_boardnet/model.py`

## Modules

`AuxiliaryReconstructionBoardNet` accepts the project's `(B, 18, 8, 8)`
board tensor only. CRTK / source / engine / verification metadata is
reporting-only and is not consumed.

1. **Stem.** A `3x3` `Conv2d(input_channels -> encoder_width)` followed
   by `BatchNorm2d` and `ReLU` lifts the board planes into a working
   channel dimension while preserving the `8 x 8` spatial layout.
2. **Encoder trunk.** `encoder_depth` `_ResidualBlock` units (two
   `3x3` `Conv2d` layers with `BatchNorm2d`, `ReLU`, and `Dropout2d`)
   refine the latent feature map. Spatial size stays at `8 x 8` so the
   decoder can reconstruct each input plane square-by-square.
3. **Classifier head.** `AdaptiveAvgPool2d -> Flatten ->
   Linear(encoder_width, hidden_dim) -> ReLU -> Dropout ->
   Linear(hidden_dim, 1)` produces the one-logit puzzle output.
4. **Reconstruction decoder.** `Conv2d(encoder_width, decoder_width,
   3x3) -> BatchNorm2d -> ReLU -> Dropout2d -> Conv2d(decoder_width,
   R, 1x1)` produces per-square logits over the `R = len(targets)`
   reconstructed planes. The decoder reads from the same latent feature
   map as the classifier, so its gradients regularise the encoder.
5. **Plane selection.** `reconstruction_targets` lists which input plane
   indices are reconstructed. The default reconstructs all 18 simple_18
   planes (12 piece occupancies, side-to-move, four castling flags, and
   the en-passant square), all of which are part of the current board
   and contain no future or engine information.

## Loss

Default trainer wiring uses the standard BCE-with-logits on
`output["logits"]`. `auxiliary_reconstruction_loss(output, target,
lambda_recon)` exposes the combined classifier + reconstruction
objective for ablations:

```
loss = BCE(logits, y) + lambda_recon * BCE(reconstruction_logits,
                                            x[:, reconstruction_targets])
```

`lambda_recon` defaults to `0.05`. Both terms use BCE-with-logits because
the reconstructed planes are binary indicators (occupancy, side-to-move,
castling, and en-passant), matching the packet's recommended loss
formulation.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)`. BCE-compatible puzzle log-odds for the
  one-logit `puzzle_binary` head.
- `logit`, `prob`: aliases of the puzzle log-odds and probability.
- `latent`: shape `(B, encoder_width, 8, 8)`, the shared feature map
  feeding both heads.
- `reconstruction_logits`: shape `(B, R, 8, 8)`, per-square reconstruction
  logits, one channel per selected target plane.
- `reconstruction_probs`: shape `(B, R, 8, 8)`, sigmoid of the logits.
- `reconstruction_target_planes`: shape `(B, R, 8, 8)`, the input slice
  used as the reconstruction target.
- `reconstruction_target_indices`: long tensor of selected plane indices.
- `reconstruction_error`: shape `(B,)`, mean-squared error between the
  reconstructed probability map and the target planes per example.
- `reconstruction_bce_per_plane`: shape `(B, R)`, per-plane BCE
  diagnostic for ablation reporting.
- `mechanism_energy`, `proposal_profile_strength`,
  `proposal_keyword_count`: scalars preserved for compatibility with the
  project's research-packet diagnostic schema.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. Engine, verification, source,
  CRTK, principal-variation, mate-score, and best-move metadata is
  reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed
  above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine
  label `2` maps to binary target `1`.
- Reconstruction targets are drawn from the current board only, which is
  the same information the classifier consumes; no future, engine, or
  CRTK signal is reconstructed.
