# Architecture

`One-Ply Counterfactual Move Landscape Network` implements CML-Net as a board-only model. It classifies puzzle-likeness from the shape of the side-to-move pseudo-legal one-ply consequence set, not from engine analysis or a move tree.

## Implementation Binding

- Registered model name: `one_ply_counterfactual_move_landscape_network`
- Source implementation file: `src/chess_nn_playground/models/move_landscape_net.py`
- Idea-local wrapper: `ideas/i025_one_ply_counterfactual_move_landscape_network/model.py`

## Components

- `Simple18BoardAdapter`: decodes piece occupancy, side to move, castling planes, and en-passant target plane from the current-board `simple_18` tensor. The deterministic move enumerator fails closed for unsupported encodings.
- `PseudoLegalDeltaEnumerator`: enumerates side-to-move pseudo-legal pawn, knight, bishop, rook, queen, king, promotion, en-passant, and optional geometric castling candidates. It does not filter by check, detect mate or stalemate, call an engine, or inspect labels.
- `RootBoardEncoder`: applies a compact convolutional root stem to the full board tensor and produces both square features and a global root embedding.
- `MoveRecordEncoder`: gathers root features from each move source and destination, uses the destination-minus-source feature delta, and adds embeddings for moving piece, capture, promotion, special move type, and relative displacement.
- `LandscapeSetPool`: computes a shared learned move energy, masked softmax attention, mean and variance move-set pools, entropic free-energy gap, top-2 energy gap, normalized entropy, and attention peak.
- `MoveLandscapeNet`: concatenates root state and permutation-invariant landscape pools, then returns one puzzle logit plus diagnostics.

## Forward Contract

```text
output = model(x)
x.shape == (batch, input_channels, 8, 8)
output["logits"].shape == (batch,)
```

The diagnostic output includes `move_landscape_free_energy`, `move_landscape_entropy`, `move_energy_mean`, `move_energy_max`, `move_energy_top2_gap`, `move_attention_peak`, `pseudo_legal_move_count`, `capture_move_fraction`, and `promotion_move_fraction`.

## Permutation Invariance

The enumerator uses a deterministic order only for reproducible padding. The network encodes each move with shared weights and then pools by masked mean, variance, softmax-weighted sum, log-sum-exp free-energy, entropy, and top-2 gap. These operations are invariant to reordering the valid move slots, so move-list order is not a learned feature.
