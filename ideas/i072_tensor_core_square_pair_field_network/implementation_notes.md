# Implementation Notes

- Central code: `src/chess_nn_playground/models/tensor_core_square_pair_field.py`.
- Registry key: `tensor_core_square_pair_field_network`.
- Idea wrapper: `ideas/i072_tensor_core_square_pair_field_network/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2148_friday_shanghai_tensorcore_pairfield.md`.
- Input is board-only `simple_18`; engine outputs, source metadata, verification
  fields, and CRTK provenance are not consumed as model inputs.
- The first config uses a medium dense pair-field shape: `model_dim: 128`,
  `heads: 8`, `head_dim: 32`, `layers: 3`, and `pair_rank: 16`.
- Forward-pass dense work is expressed with `einsum`/matmul-style operations over
  fixed `(64, 64)` pair tensors. The implementation loops over layers but not over
  heads or squares.
- The classifier emits one BCE puzzle logit because the repository's
  `puzzle_binary` trainer expects shape `(B,)` for `num_classes: 1`.
