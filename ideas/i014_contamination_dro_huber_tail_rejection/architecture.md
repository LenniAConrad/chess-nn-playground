# Architecture

The model is a compact convolutional board encoder with global mean/max pooling and low-order material statistics. It returns one binary puzzle logit and diagnostics; all distinctive behavior is in the training objective.

## Implementation Binding

- Registered model name: `contamination_dro_huber_tail_rejection`.
- Source implementation: `src/chess_nn_playground/models/gpt_research_architectures.py`.
- Idea-local wrapper: `ideas/i014_contamination_dro_huber_tail_rejection/model.py`.
