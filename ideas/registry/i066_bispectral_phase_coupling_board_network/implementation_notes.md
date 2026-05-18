# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/bispectral_phase_coupling.py`.
- Idea-local wrapper: `ideas/registry/i066_bispectral_phase_coupling_board_network/model.py`.
- Registry key: `bispectral_phase_coupling_board_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2110_friday_shanghai_bispectral_phase.md`.
- Board-only: the architecture does not consume engine, verification, source, or CRTK metadata as input.
- The 2D FFT and selected `(k, l)` bispectral / power / cross-channel frequency tables are stored as non-learnable buffers; only the `1x1` channel mixer, `LayerNorm`, and head linears are trainable.
- Ablation modes (`ablation`): `none` (default), `magnitude_only`, `power_only`, `phase_batch_shuffle`, `random_frequency_pairs`, `channel_pair_shuffle`, `no_coordinate_planes`.
- Default feature dimension at `simple_18` with `Cmix=16`, `T=48`, 16 power frequencies, 8 channel pairs, and 12 cross frequencies: `2868`.
