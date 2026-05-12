# Architecture

`Multi-Scale Dilated Board Mixer CNN` is a compact conventional chess
CNN whose every block sees several chess-relevant spatial ranges at
once. Each block runs four parallel branches in lockstep:

- 3x3 convolution with dilation 1 (adjacent local patterns),
- 3x3 convolution with dilation 2 (knight-distance / short diagonal),
- 3x3 convolution with dilation 3 (longer board relations),
- 1x1 channel-only branch.

The branches are concatenated along channels and fused by a 1x1 mixer
projection back to ``width`` with a residual connection and BatchNorm.
Four fixed coordinate planes are appended to the simple_18 input
(normalized rank, normalized file, center-distance, side-to-move-
relative forward direction) so the trunk has explicit board geometry
without any engine or CRTK metadata. After the trunk a global context
gate uses mean+max pooling and a small MLP to emit a sigmoid channel
scale that modulates the trunk activations. The head pools mean+max
over the gated 8x8 features and returns ``num_classes`` logits.

Tensor contract:

```text
input:                    (B, 18, 8, 8)
with coordinate planes:   (B, 22, 8, 8)
stem:                     (B, width, 8, 8)
each mixer block:         (B, width, 8, 8)
global context gate:      (B, width, 8, 8)
pooled head:              (B, num_classes)
```

For ``num_classes == 1`` the head logits are squeezed to ``(B,)`` so
the puzzle-binary trainer can drive ``BCEWithLogitsLoss`` directly.

Forward pass returns a dict so the trainer can pull diagnostics and
the puzzle logit from the same call:

- ``logits`` (puzzle logit, shape ``(B,)``),
- ``two_class_logits`` (shape ``(B, 2)`` derived as
  ``[-0.5 * logits, +0.5 * logits]`` when ``num_classes == 1``),
- ``stem_energy``, ``trunk_energy``, ``coord_plane_energy``
  (mean square magnitudes for the stem, post-trunk and the appended
  coordinate channels),
- ``context_gate_mean / std / min / max`` (statistics of the
  per-channel sigmoid gate vector),
- ``branch_count`` (active parallel branches per block; the markdown
  default is 4: dilation 1, 2, 3 and the 1x1),
- ``active_dilations`` (count of distinct 3x3 dilation rates active
  in each block),
- ``ablation_active`` (1.0 when a section-6 ablation is enabled, else
  0.0).

## Section 6 Ablations

The model exposes the markdown's section 6 ablation table via
``model.ablation`` so the same config can drive the central
falsifier and every supporting falsifier:

| Ablation | What it removes / changes |
|---|---|
| ``"none"`` | Default multi-scale model. |
| ``"single_dilation_matched"`` | Replace every parallel branch with a single 3x3 dilation-1 branch at matched 4x branch_width capacity. |
| ``"no_dilation_3"`` | Drop the dilation-3 branch but keep dilations 1, 2 and the 1x1. |
| ``"no_coordinate_planes"`` | Skip the appended rank/file/center/forward planes; the stem sees the raw (B, 18, 8, 8) board. |
| ``"no_global_context_gate"`` | Skip the channel gate so the trunk is pooled directly. |
| ``"small_width_control"`` | Run the same multi-scale architecture at half width. |
| ``"residual_cnn_matched_params"`` | Replace the multi-scale trunk with a plain residual-CNN trunk at matched parameter count. |

## Implementation Binding

- Registered model name: ``multi_scale_dilated_board_mixer_cnn``.
- Source implementation file:
  ``src/chess_nn_playground/models/multi_scale_dilated_board_mixer_cnn.py``
  (defines ``BoardCoordinatePlanes``,
  ``MultiScaleDilatedMixerBlock``, ``GlobalContextGate``,
  ``MultiScaleHead``, ``ResidualCNNControl`` and
  ``MultiScaleBoardMixerCNN``, plus
  ``build_multi_scale_dilated_board_mixer_cnn_from_config``).
- Idea-local wrapper:
  ``ideas/registry/i064_multi_scale_dilated_board_mixer_cnn/model.py``
  exports ``build_model_from_config`` which calls
  ``build_multi_scale_dilated_board_mixer_cnn_from_config`` after
  defaulting ``num_classes`` to 1 for the puzzle-binary trainer.
- Inputs are board-only; engine, source, verification and CRTK
  metadata are never used as model input.
