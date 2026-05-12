# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/ray_state_space_scan.py`.
- Idea-local wrapper: `ideas/registry/i125_ray_state_space_scan_network/model.py`.
- Registry key: `ray_state_space_scan_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.
- Batch candidate: `Ray State-Space Scan Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Bespoke implementation. The shared `ResearchPacketProbe` wrapper is no longer used.
