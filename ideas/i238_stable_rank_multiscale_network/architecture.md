# Architecture

`Stable-Rank Multiscale Network` is a hand-coded `nn.Module` at
`src/chess_nn_playground/models/stable_rank_multiscale.py`.

- Mechanism family: `linear_algebra` (bespoke).
- Module class: `StableRankMultiscaleNetwork`.
- Registry name: `stable_rank_multiscale_network`.
- Input: board tensor `(B, 18, 8, 8)`.
- Output: `(B, num_classes)` logits (squeezed when num_classes=1).
- Compute: One svdvals top-1 per scale (3 calls), differentiable through bilinear interaction.

See the source packet for the full mathematical derivation and the
module file for the exact algebraic operator implementation.

## Implementation Binding

- Registered model name: `stable_rank_multiscale_network`.
- Source implementation: `src/chess_nn_playground/models/stable_rank_multiscale.py`.
- Idea-local wrapper: `ideas/i238_stable_rank_multiscale_network/model.py`.
