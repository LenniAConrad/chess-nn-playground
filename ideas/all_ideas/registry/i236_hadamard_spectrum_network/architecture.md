# Architecture

`Hadamard Walsh-Spectrum Network` is a hand-coded `nn.Module` at
`src/chess_nn_playground/models/hadamard_spectrum.py`.

- Mechanism family: `linear_algebra` (bespoke).
- Module class: `HadamardSpectrumNetwork`.
- Registry name: `hadamard_spectrum_network`.
- Input: board tensor `(B, 18, 8, 8)`.
- Output: `(B, num_classes)` logits (squeezed when num_classes=1).
- Compute: Fixed orthogonal Walsh basis (no parameters) + tiny channel-mix conv + MLP. Cost dominated by the einsum WHT.

See the source packet for the full mathematical derivation and the
module file for the exact algebraic operator implementation.

## Implementation Binding

- Registered model name: `hadamard_spectrum_network`.
- Source implementation: `src/chess_nn_playground/models/hadamard_spectrum.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i236_hadamard_spectrum_network/model.py`.
