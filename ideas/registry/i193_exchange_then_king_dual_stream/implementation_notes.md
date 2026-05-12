# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/i193_exchange_then_king_dual_stream/model.py`.
- Registry key: `exchange_then_king_dual_stream`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Exchange-Then-King Dual Stream`.
- Bespoke board-only architecture; CRTK / engine / verification metadata is reporting-only and never used as model input.
- Closed-form deterministic feature builder constructs the per-stream feature stacks from precomputed geometric attack and between-square tables; the encoder, heads, and phase router are the only learned modules.
- Supported ablations: `none`, `shared_stream_only`, `fixed_half_gate`, `king_only`, `exchange_only`.
