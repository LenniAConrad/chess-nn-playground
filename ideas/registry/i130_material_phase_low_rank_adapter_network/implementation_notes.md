# Implementation Notes

- Bespoke model: `src/chess_nn_playground/models/trunk/material_phase_low_rank_adapter.py`.
- Idea-local wrapper: `ideas/registry/i130_material_phase_low_rank_adapter_network/model.py`.
- Registry key: `material_phase_low_rank_adapter_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2133_friday_shanghai_architecture_batch_6.md`.
- Batch candidate: `Material-Phase Low-Rank Adapter Network`.
- Board-only model: the deterministic material/phase summary is computed from the simple_18 input planes; no engine, verification, source, or CRTK metadata is consumed as model input.
- Adapter rank defaults to `4`; `B(s)` is zero-initialised so training starts at the shared backbone.
