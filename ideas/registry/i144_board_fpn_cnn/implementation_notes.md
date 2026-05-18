# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/board_fpn_cnn.py`.
- Idea-local wrapper: `ideas/registry/i144_board_fpn_cnn/model.py`.
- Registry key: `board_fpn_cnn`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.
- Board-only: the architecture does not consume engine, verification, source, or CRTK metadata as input.
- Ablation modes (`ablation`): `none` (default), `single_resolution_matched`, `bottom_up_only`, `no_2x2_level`, `late_pool_only`, `no_coordinate_planes`.
- Config keys (with backwards-compatible aliases): `width` (alias `channels`), `blocks_per_level` (alias `depth`), `hidden_dim`, `dropout`, `use_batchnorm`, `use_coordinate_planes`, `ablation`.
- Three convolutional stacks at `8x8`, `4x4`, `2x2`. Top-down 1x1 projections `project2_to4` and `project4_to8` reuse a single bottom-up sweep.
- Head pool dimension at width `W`: `14 * W` features.
