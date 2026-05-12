# Architecture

The model has a weak statistics prior, a convolutional posterior probe, a straight-through binary concrete gate, and a gate-only classifier head.

## Implementation Binding

- Registered model name: `conditional_surprisal_gate`.
- Source implementation: `src/chess_nn_playground/models/gpt_research_architectures.py`.
- Idea-local wrapper: `ideas/registry/i017_conditional_surprisal_gate/model.py`.
