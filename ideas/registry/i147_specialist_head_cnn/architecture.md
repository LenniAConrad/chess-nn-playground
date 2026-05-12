# Architecture

`Specialist-Head CNN` is a board-only convolutional classifier with a shared
trunk and several small specialist heads. It keeps the packet's plain-CNN
premise: there is no attention router or sparse mixture-of-experts layer.

## Shared Trunk

The model accepts the repository board tensor contract `B x 18 x 8 x 8`.
A residual CNN trunk maps the board to:

```text
h = CNN(x)  # (B, trunk_width, 8, 8)
```

The default config uses `trunk_width: 64` and `trunk_depth: 4`. The first block
projects the board planes to the trunk width; later blocks are residual
two-convolution blocks.

## Specialist Features

The heads use fixed, interpretable feature pools:

- `global_feat`: mean and max pooling over the full board.
- `center_feat`: mean and max pooling over a fixed `4 x 4` center mask.
- `edge_feat`: mean and max pooling over the board edge ring.
- `king_feat`: own and opponent `3 x 3` king-zone pools, decoded from the
  simple_18 king planes and side-to-move plane.
- `material_feat`: safe piece counts, count differences, material balance,
  phase, side to move, castling planes, and en-passant presence.

The king head fails closed: if exactly one white king and one black king cannot
be decoded from the board planes, its feature vector and logit are forced to
zero for that sample.

## Heads And Fusion

Each specialist has a small MLP:

```text
head_i(feat_i) -> head_feature_i, logit_i
```

The main prediction uses the packet's learned fusion sketch:

```text
fusion_input = concat(head_features, logits_i)
logits = MLP(fusion_input)
```

The output is one BCE-compatible puzzle logit for `puzzle_binary`. Diagnostics
include specialist logits, active head count, logit shares, feature energies,
king-zone decode status, king-zone masses, material balance, material phase,
and total piece count.

## Ablations

- `single_global_head`: use only the global pooling head.
- `no_king_head`: remove the king-zone specialist.
- `no_material_head`: remove the material/count specialist.
- `uniform_logit_average`: average active specialist logits without the learned
  fusion MLP.
- `same_region_random_masks`: replace center and edge masks with deterministic
  random masks of the same sizes.

## Implementation Binding

- Registered model name: `specialist_head_cnn`
- Source implementation file: `src/chess_nn_playground/models/trunk/specialist_head_cnn.py`
- Idea-local wrapper: `ideas/registry/i147_specialist_head_cnn/model.py`
