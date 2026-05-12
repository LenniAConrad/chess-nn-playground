# Implementation Notes

- Source implementation: `src/chess_nn_playground/models/slot_attention_role_binding_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i105_slot_attention_role_binding_network/model.py`.
- Registry key: `slot_attention_role_binding_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.
- Batch candidate: `Slot Attention Role Binding Network`.
- This is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input. Up to 32 occupied piece tokens are
  selected in deterministic rank-major order; padded slots are masked from
  every slot-attention key/value projection.
