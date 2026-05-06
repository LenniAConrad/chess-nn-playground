# Architecture

SRPA uses the current-board tensor only.

- Input: `lc0_bt4_112` by default.
- Square stem: small convolutional square encoder producing one embedding per board square.
- Fixed relation set: deterministic ordered square-pair tokens for queen-like rays, knight jumps, king-neighbor edges, and pawn diagonals. Long-ray relation tokens include intermediate path summaries.
- Relation projection: source square, destination square, source-destination interactions, geometry embeddings, and path embeddings are projected into relation-token space.
- Sparse bottleneck: two equal-capacity `GroupSparsePursuit` modules, one background and one tactical, each with an explicit normalized dictionary decoder.
- Pursuit: unrolled LISTA-style gradient step plus atom-wise soft threshold and group soft threshold.
- Classifier: receives only sparse residual traces, group energies, activity fractions, entropy, and residual asymmetry. No dense board or relation embedding bypass is present.
- Output: one primary logit and one auxiliary residual-asymmetry logit.

The architecture is implemented in `src/chess_nn_playground/models/sparse_relation_pursuit.py` and registered as `sparse_relation_pursuit_asymmetry`.

## Implementation Binding

- Registered model name: `sparse_relation_pursuit_asymmetry`.
- Source implementation: `src/chess_nn_playground/models/sparse_relation_pursuit.py`.
- Idea-local wrapper: `ideas/i013_sparse_relation_pursuit_asymmetry/model.py`.
