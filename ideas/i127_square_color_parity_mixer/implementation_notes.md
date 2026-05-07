# Implementation Notes

- Central code: `src/chess_nn_playground/models/square_color_parity_mixer.py`.
- Registry key: `square_color_parity_mixer`.
- Idea wrapper: `ideas/i127_square_color_parity_mixer/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.
- Batch candidate: `Square-Color Parity Mixer`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.

The implementation builds one token per board square, partitions tokens into fixed dark/light sets, then applies learned same-color and cross-color block mixers. The cross-color mixer is reused transposed for the reverse direction, matching the packet's block-matrix formulation.

Piece-type gates are computed from the `simple_18` piece planes. Bishops are initialized toward same-color flow, knights toward cross-color flow, and pawns, queens, and kings toward mixed flow. These priors are trainable, so the model can adapt gate strengths during normal puzzle-binary training.

The forward pass returns one puzzle logit plus diagnostics for gate usage and parity-block energy. Ablation variants from the packet, such as ordinary token mixing or random bipartition, are experiment variants rather than runtime branches in the production model.
