# Implementation Notes

- Source code: `src/chess_nn_playground/models/variational_board_action.py`.
- Registry key: `variational_board_action_network`.
- Idea wrapper: `ideas/all_ideas/registry/i071_variational_board_action_network/model.py`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2146_friday_shanghai_variational_board_action.md`.
- The model is board-only and does not consume engine, verification, source, CRTK, or
  provenance metadata as input.
- The repo trainer expects one BCE logit for `puzzle_binary`; the model returns
  `{"logits": tensor(B), ...diagnostics...}`.
- Exact potential autograd is not enabled in this first implementation. The configured
  path uses the packet's force-head approximation and exposes `force_head_only` as a
  semantics-control ablation.
