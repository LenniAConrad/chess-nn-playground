# Implementation Notes

- Source code: `src/chess_nn_playground/models/trunk/relational_query_algebra.py`.
- Registry key: `relational_query_algebra_network`.
- Idea wrapper: `ideas/registry/i070_relational_query_algebra_network/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2139_friday_shanghai_relational_query_algebra.md`.
- The model is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input.
- The repo trainer expects one BCE logit for `puzzle_binary`; the model returns
  `{"logits": tensor(B), ...diagnostics...}`.
