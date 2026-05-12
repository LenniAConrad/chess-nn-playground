# Implementation Notes

- Central code: `src/chess_nn_playground/models/neural_board_cellular_automaton.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i115_neural_board_cellular_automaton/model.py` (`build_model_from_config`).
- Registry key: `neural_board_cellular_automaton`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.
- Batch candidate: `Neural Board Cellular Automaton`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The local update rule's weights are **tied across all CA steps**; this is the load-bearing structural difference from a stacked residual CNN. Untying weights or removing the iteration loop changes the model.
- The 1x1 output of the update rule is zero-initialized so the untrained network has a stable identity fixed point.
