# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/set_query_attention.py`.
- Registry key: `set_query_attention_bottleneck`.
- Idea wrapper: `ideas/registry/i102_set_query_attention_bottleneck/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.
- Batch candidate: `Set-Query Attention Bottleneck`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The implementation builds 64 square tokens, applies a learned query bank over
  projected token keys and values, exports attention maps and per-query
  diagnostics, and supports the source packet ablations.
