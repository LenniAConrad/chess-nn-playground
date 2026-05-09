# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/convex_feasibility.py` (`ConvexFeasibilityResidualNetwork` and `build_convex_feasibility_residual_network_from_config`).
- Idea-local wrapper: `ideas/i094_convex_feasibility_residual_network/model.py` calls the bespoke builder; it does not depend on the shared `ResearchPacketProbe` scaffold.
- Registry key: `convex_feasibility_residual_network` (registered in `src/chess_nn_playground/models/registry.py`; removed from `RESEARCH_PACKET_MODEL_NAMES`).
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.
- Batch candidate: `Convex Feasibility Residual Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Ablation modes: `projection` (default), `no_projection`, `random_constraints`, `linear_head_same_params`, `material_only_encoder`. These keep parameter counts comparable so any metric delta is attributable to the feasibility mechanism rather than capacity.
