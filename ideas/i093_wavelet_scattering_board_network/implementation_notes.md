# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/wavelet_scattering_board_network.py`.
- Idea-local wrapper: `ideas/i093_wavelet_scattering_board_network/model.py`.
- Registry key: `wavelet_scattering_board_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.
- Batch candidate: `Wavelet Scattering Board Network`.
- Board-only: the architecture does not consume engine, verification, source, or CRTK metadata as input.
- Wavelet filters are stored as non-learnable replicated buffers; only the `LayerNorm` and head linears are trainable.
- Ablation modes (`mode`): `haar` (default), `random_fixed_filters`, `lowpass_only`, `channel_shuffle`.
- Default feature dimension at `simple_18` with three scales and second-order scattering: `1026`.
