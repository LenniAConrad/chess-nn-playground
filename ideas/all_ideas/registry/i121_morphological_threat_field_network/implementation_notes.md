# Implementation Notes

- Central code: `src/chess_nn_playground/models/morphological_threat_field_network.py`
  (class `MorphologicalThreatFieldNetwork`, builder
  `build_morphological_threat_field_network_from_config`).
- Idea-local wrapper: `ideas/all_ideas/registry/i121_morphological_threat_field_network/model.py`
  exposes `build_model_from_config` for the trainer.
- Registry key: `morphological_threat_field_network` (registered in
  `src/chess_nn_playground/models/registry.py`).
- The model name is excluded from `RESEARCH_PACKET_MODEL_NAMES` so the
  implementation-kind audit detects this folder as `bespoke_model`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.
- Batch candidate: `Morphological Threat Field Network`.
- This is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input.
