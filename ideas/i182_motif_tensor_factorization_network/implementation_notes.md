# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/motif_tensor_factorization_network.py`
  (`MotifTensorFactorizationNetwork`).
- Registry key: `motif_tensor_factorization_network`.
- Idea-local wrapper: `ideas/i182_motif_tensor_factorization_network/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.
- Batch candidate: `Motif Tensor Factorization Network` (rank 6).
- Input contract: `simple_18` current-board tensor `(B, 18, 8, 8)`.
- Output contract: `dict` whose `"logits"` is the puzzle logit fed to
  BCE-with-logits.
- The implementation is intentionally board-only and does not consume
  engine, verification, source, or CRTK metadata as input.
