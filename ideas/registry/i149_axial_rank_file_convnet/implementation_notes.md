# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/axial_rank_file_convnet.py`.
- Idea-local wrapper: `ideas/registry/i149_axial_rank_file_convnet/model.py`.
- Registry key: `axial_rank_file_convnet`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.
- Board-only: the architecture does not consume engine, verification, source, or CRTK metadata as input.
- Ablation modes (`ablation`): `none` (default), `local_only`, `rank_only`, `file_only`, `no_residual`, `single_block`.
- Config keys: `channels`, `depth` (alias `blocks`), `hidden_dim`, `dropout`, `use_batchnorm`, `ablation`.
- Rank-wise 1D conv has spatial kernel `(1, 8)`; file-wise 1D conv has spatial kernel `(8, 1)`. Outputs are truncated back to `8x8`.
- Head pool dimension at width `C`: `34 * C` features.
