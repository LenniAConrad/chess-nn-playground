# Implementation Notes

- Central code: `src/chess_nn_playground/models/tactical_state_bottleneck.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i091_tactical_state_bottleneck_inference/model.py`.
- Registry key: `tactical_state_bottleneck_inference`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-28_0901_tuesday_new_york_tactical_latent.md`.
- The model is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The shared puzzle_binary trainer calls `model(board)` and trains through the prior-path BCE on `output["logits"]`. The full posterior-aware multi-loss objective (`forward_train` + `tactical_state_loss_components`) is exposed for ablation runs that drive the model directly.
- `NoLatentMatchedBaseline` lives next to the main module and provides the matched no-latent control required by the packet's ablation programme.
