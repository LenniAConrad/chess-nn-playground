# Implementation Notes

- Central code: `src/chess_nn_playground/models/neural_decision_forest_boardnet.py`.
- Idea-local wrapper: `ideas/registry/i158_neural_decision_forest_boardnet/model.py`.
- Registry key: `neural_decision_forest_boardnet`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.
- Batch candidate: `Neural Decision Forest BoardNet`.
- Architecture is board-only and consumes no engine, verification,
  source, or CRTK metadata as input.
- The convolutional trunk is paired with a fully soft differentiable
  decision forest: a single shared linear layer produces every internal
  node's oblique split logit, and leaf path probabilities are computed
  in closed form by indexing precomputed `path_nodes` /
  `path_directions` buffers along the tree depth.
- The forest is fully differentiable (no sparse routing). Every leaf
  receives gradient on every sample weighted by its soft routing
  probability `mu_{t, ell}(z)`.
- `tree_depth` is bounded to `<= 8` to keep the path tensors compact;
  `num_trees` defaults to 8 and `tree_depth` defaults to 4 for the
  paper-grade `puzzle_binary` config.
