# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/tensorsketch_interaction_network.py`.
- Idea-local wrapper: `ideas/i108_tensorsketch_interaction_network/model.py` (calls
  `build_tensorsketch_interaction_network_from_config` directly; no `ResearchPacketProbe`).
- Registry key: `tensorsketch_interaction_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.
- Batch candidate: `TensorSketch Interaction Network`.
- The model is intentionally board-only: the feature extractor consumes the
  `simple_18` tensor and reduces global planes to scalars (max). It does not
  consume engine, verification, source, or CRTK metadata.
- CountSketch hashes/signs are sampled once from `sketch_seed` and stored as
  `state_dict` buffers so the kernel approximation is deterministic.
- The forward pass returns a dict whose `logits` entry has shape `(B,)` to
  match the puzzle_binary BCE-with-logits trainer contract.
