# Architecture

`Möbius Piece-Constellation Network` implements the MPCN packet as a low-rank polynomial set functional over occupied current-board piece-square facts. It uses no convolutional trunk, residual spatial stack, transformer attention, graph message passing, legal move generation, attack graph, sheaf, or transport operator.

## Architecture

Input `x` has shape `[B, 18, 8, 8]` for the supported `simple_18` path. `SafeBoardStateAdapter` validates the board tensor, extracts piece planes `0:12`, and flattens the safe state planes `12:18` containing side-to-move, castling rights, and en-passant state. Non-`simple_18` encodings fail closed unless an explicit current-board piece-plane `channel_map` is provided.

`PieceSquareTokenizer` turns the 12 piece planes into 64 occupied square token vectors without using argmax or tuple enumeration. Each token combines a learned piece embedding, a learned square embedding, and a learned joint piece-square table. Empty squares are multiplied by an occupancy mask and therefore contribute zero token vectors.

`ElementarySymmetricInteractionBlock` computes the degree-isolated Möbius/ANOVA features by the descending elementary-symmetric recurrence:

```text
E3 <- E3 + E2 * v
E2 <- E2 + E1 * v
E1 <- E1 + v
```

The descending update prevents a token from interacting with itself. The model keeps `H1`, `H2`, and `H3`, normalized by `sqrt(n)`, `sqrt(C(n,2))`, and `sqrt(C(n,3))` when `normalize_by_tuple_count` is enabled.

`DegreeGate` applies learned sigmoid gates to each degree channel. `ConstellationClassifierHead` layer-normalizes the gated degree features, concatenates them with the safe state embedding, and maps them through a compact MLP to the configured logits. With `num_classes: 1`, the repo puzzle-binary trainer receives one logit of shape `[B]` for fine label `2` versus labels `0` and `1`. With `num_classes: 2`, the same head returns two logits of shape `[B, 2]`.

Returned diagnostics include degree norms, occupied count, mean occupancy, degree-gate summaries, state embedding norm, and the optional gate sparsity auxiliary value.

## Implementation Binding

- Registered model name: `mobius_piece_constellation_network`
- Source implementation file: `src/chess_nn_playground/models/mobius_piece_constellation.py`
- Idea-local wrapper: `ideas/i037_mobius_piece_constellation_network/model.py`
