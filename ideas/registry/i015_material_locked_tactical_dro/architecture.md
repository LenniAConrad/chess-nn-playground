# Architecture

The model combines a compact board encoder with deterministic tactical masks. A bounded contamination branch produces an adversarial diagnostic logit used only by the loss.

## Implementation Binding

- Registered model name: `material_locked_tactical_dro`.
- Source implementation: `src/chess_nn_playground/models/trunk/gpt_research_architectures.py`.
- Idea-local wrapper: `ideas/registry/i015_material_locked_tactical_dro/model.py`.
