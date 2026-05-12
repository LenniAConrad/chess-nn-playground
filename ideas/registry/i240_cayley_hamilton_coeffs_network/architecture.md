# Architecture

`Cayley-Hamilton Coefficient Network` is a hand-coded `nn.Module` at
`src/chess_nn_playground/models/trunk/cayley_hamilton_coeffs.py`.

- Mechanism family: `linear_algebra` (bespoke).
- Module class: `CayleyHamiltonCoefficientNetwork`.
- Registry name: `cayley_hamilton_coeffs_network`.
- Input: board tensor `(B, 18, 8, 8)`.
- Output: `(B, num_classes)` logits (squeezed when num_classes=1).
- Compute: r matmul iterations of Faddeev-LeVerrier at r=12; fully differentiable; auxiliary Cayley-Hamilton residual ||A^r + sum c_k A^{r-k}||_F as sanity feature.

See the source packet for the full mathematical derivation and the
module file for the exact algebraic operator implementation.

## Implementation Binding

- Registered model name: `cayley_hamilton_coeffs_network`.
- Source implementation: `src/chess_nn_playground/models/trunk/cayley_hamilton_coeffs.py`.
- Idea-local wrapper: `ideas/registry/i240_cayley_hamilton_coeffs_network/model.py`.
