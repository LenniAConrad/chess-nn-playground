# Architecture

The model is intentionally plain. Novelty is the loss: BCE plus soft odd-even sort residual on logits and binary targets.

## Implementation Binding

- Registered model name: `soft_sorting_order_residual_ranker`.
- Source implementation: `src/chess_nn_playground/models/gpt_research_architectures.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i016_soft_sorting_order_residual_ranker/model.py`.
