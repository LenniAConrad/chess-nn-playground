# Architecture

`Empty-Square Opportunity Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repo's simple 18-plane current-board tensor
with shape `(B, 18, 8, 8)` and returns one puzzle logit per position.

The model builds occupied and empty masks directly from the first twelve piece
planes:

```text
occ_mask = any_piece_plane(board)
empty_mask = 1 - occ_mask
```

The shared trunk receives the board tensor plus optional rank/file coordinate
planes:

```text
h = CNNStem(concat(board, rank_plane, file_plane))
```

The occupied branch reads only occupied-square trunk activations:

```text
h_occ = ConvStack(h * occ_mask)
z_occ_raw = mean/max/topk_pool(h_occ, mask=occ_mask)
z_occ = Linear(z_occ_raw)
```

The empty branch reads only empty-square trunk activations and emits learned
opportunity maps:

```text
h_empty = ConvStack(h * empty_mask)
opportunity = Conv1x1(h_empty -> opportunity_channels)
z_empty_raw = mean/max/topk_pool(opportunity, mask=empty_mask)
z_empty = Linear(z_empty_raw)
```

The classifier fuses occupied evidence, empty opportunity evidence, and explicit
interaction terms:

```text
z_pair = concat(z_occ, z_empty, z_occ * z_empty, abs(z_occ - z_empty))
logits = MLP(z_pair)
```

The named opportunity channels (`escape_like`, `landing_like`, `blocker_like`,
`promotion_lane_like`, and `king_zone_empty_like`) are learned diagnostics only;
they are not supervised labels. The implementation also supports the packet
ablations `occupied_only`, `empty_only`, `random_empty_mask`,
`no_occ_empty_interaction`, and `cnn_matched_params`.

Diagnostics include opportunity maps, empty opportunity norm, occupied and empty
branch norms, interaction energy, top opportunity square, occupancy count, empty
count, and named opportunity-channel means.

## Implementation Binding

- Registered model name: `empty_square_opportunity_network`
- Source implementation file: `src/chess_nn_playground/models/empty_square_opportunity_network.py`
- Idea-local wrapper: `ideas/i162_empty_square_opportunity_network/model.py`
