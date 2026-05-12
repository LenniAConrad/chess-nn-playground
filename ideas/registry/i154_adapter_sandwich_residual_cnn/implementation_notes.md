# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/adapter_sandwich_residual_cnn.py`.
- Idea-local wrapper: `ideas/registry/i154_adapter_sandwich_residual_cnn/model.py`.
- Registry key: `adapter_sandwich_residual_cnn`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.
- Batch candidate: `Adapter-Sandwich Residual CNN`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- `adapter_dim` defaults to `max(4, channels // 4)`; override via the `adapter_dim` config key if you want to test a different bottleneck width.
- Adapter `W_up` weights are zero-initialised so the network behaves identically to a plain residual CNN at the first training step. The adapters earn non-zero contribution only as training proceeds.
