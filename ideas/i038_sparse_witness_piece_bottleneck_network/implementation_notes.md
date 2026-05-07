# Implementation Notes

- Central code: `src/chess_nn_playground/models/sparse_witness_bottleneck.py`.
- Registry key: `sparse_witness_piece_bottleneck_network`.
- Idea-local wrapper: `ideas/i038_sparse_witness_piece_bottleneck_network/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0713_tuesday_los_angeles_sparse_witness_bottleneck.md`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.

The implementation follows the packet's minimum-description bottleneck:

1. Validate board channels with `EncodingAdapter`.
2. Score current occupied piece-squares with `BoardScorer`.
3. Select `min(K, n_occupied)` witnesses using `OccupiedPieceTopKSelector`.
4. Censor all unselected current-piece planes.
5. Classify the witness grid with `WitnessGridEncoder`.

The downstream classifier receives a hard binary mask and censored piece planes only. Continuous selector probabilities are not concatenated to the classifier input.

`simple_18` is the default and uses explicit current-board semantics. Any non-`simple_18` experiment must supply explicit `adapter.piece_plane_indices` and `adapter.global_plane_indices`; otherwise model construction raises a clear error.
