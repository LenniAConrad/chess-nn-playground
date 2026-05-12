# Architecture

`Permanent Ryser Coupling Network` is a hand-coded `nn.Module` at
`src/chess_nn_playground/models/trunk/permanent_ryser.py`.

- Mechanism family: `linear_algebra` (bespoke).
- Module class: `PermanentRyserNetwork`.
- Registry name: `permanent_ryser_network`.
- Input: board tensor `(B, 18, 8, 8)`.
- Output: `(B, num_classes)` logits (squeezed when num_classes=1).
- Compute: Ryser at k=6: 64 subset iterations of inclusion-exclusion; fully differentiable.

See the source packet for the full mathematical derivation and the
module file for the exact algebraic operator implementation.

## Implementation Binding

- Registered model name: `permanent_ryser_network`.
- Source implementation: `src/chess_nn_playground/models/trunk/permanent_ryser.py`.
- Idea-local wrapper: `ideas/registry/i239_permanent_ryser_network/model.py`.
