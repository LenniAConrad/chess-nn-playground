# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/rank_quantile.py` defines `EvidenceFieldEncoder`, `RankQuantilePooler`, and `RankQuantileEvidenceFieldNetwork` plus `build_rank_quantile_evidence_field_network_from_config`.
- Idea-local wrapper: `ideas/i095_rank_quantile_evidence_field_network/model.py` exposes `build_model_from_config(config)` and forwards the `model:` block (after stripping the registry-only keys `name` / `packet_profile` / `mechanism_family`) to the bespoke builder.
- Registry key: `rank_quantile_evidence_field_network` (registered in `src/chess_nn_playground/models/registry.py` and explicitly removed from `RESEARCH_PACKET_MODEL_NAMES` so it is no longer routed through the shared `ResearchPacketProbe`).
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2048_friday_shanghai_architecture_batch_2.md`.
- Batch candidate: `Rank-Quantile Evidence Field Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Supported ablation modes (via `model.mode`): `quantile`, `mean_pool_only`, `topk_only`, `random_field_encoder`, `square_shuffle`.
