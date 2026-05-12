# Implementation Notes

- Central code: `src/chess_nn_playground/models/pawn_skeleton_barrier.py`.
- Idea-local wrapper: `ideas/registry/i126_pawn_skeleton_barrier_network/model.py` (calls `build_pawn_skeleton_barrier_network_from_config`).
- Registry key: `pawn_skeleton_barrier_network` (registered in `src/chess_nn_playground/models/registry.py`; explicitly excluded from `RESEARCH_PACKET_MODEL_NAMES`).
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.
- Batch candidate: `Pawn Skeleton Barrier Network`.
- Input is the repo `simple_18` board tensor only; no engine, verification, source, or CRTK metadata is consumed as model input.
