# Architecture

`Cayley Orthogonal Map Network` is a hand-coded `nn.Module` at
`src/chess_nn_playground/models/cayley_orthogonal.py`.

- Mechanism family: `linear_algebra` (bespoke).
- Module class: `CayleyOrthogonalNetwork`.
- Registry name: `cayley_orthogonal_network`.
- Input: board tensor `(B, 18, 8, 8)`.
- Output: `(B, num_classes)` logits (squeezed when num_classes=1).
- Compute: One torch.linalg.solve per board on r x r (r=12), spectral-clipped Frobenius keeps Q well-conditioned.

See the source packet for the full mathematical derivation and the
module file for the exact algebraic operator implementation.

## Implementation Binding

- Registered model name: `cayley_orthogonal_network`.
- Source implementation: `src/chess_nn_playground/models/cayley_orthogonal.py`.
- Idea-local wrapper: `ideas/i237_cayley_orthogonal_network/model.py`.
