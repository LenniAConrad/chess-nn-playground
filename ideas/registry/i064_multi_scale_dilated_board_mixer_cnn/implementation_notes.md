# Implementation Notes

- Central code: `src/chess_nn_playground/models/multi_scale_dilated_board_mixer_cnn.py`.
- Idea-local wrapper: `ideas/registry/i064_multi_scale_dilated_board_mixer_cnn/model.py`
  re-exports `build_model_from_config`, which calls
  `build_multi_scale_dilated_board_mixer_cnn_from_config` after defaulting
  `num_classes` to 1 for the puzzle-binary trainer.
- Registry key: `multi_scale_dilated_board_mixer_cnn`.
- Builder: `build_multi_scale_dilated_board_mixer_cnn_from_config`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2107_friday_shanghai_multiscale_cnn_mixer.md`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The repo-canonical config keys (`channels`, `hidden_dim`, `depth`)
  are mapped onto the markdown's `width`, `head_hidden`, and
  `num_blocks`. The `branch_width` field defaults to `width // 2`
  when not provided.
- Section 6 falsifier ablations are exposed via `model.ablation`:
  `none`, `single_dilation_matched`, `no_dilation_3`,
  `no_coordinate_planes`, `no_global_context_gate`,
  `small_width_control`, `residual_cnn_matched_params`.
