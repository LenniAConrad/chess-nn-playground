# Architecture

`Multiplicative Conjunction ConvNet` is a board-only convolutional classifier for
the `puzzle_binary` task. It accepts the repo's simple 18-plane current-board
tensor with shape `(B, 18, 8, 8)` and returns one puzzle logit per position.

The model starts with a plain two-layer CNN stem. When enabled, fixed rank and
file coordinate planes are appended before the stem:

```text
h = CNNStem(concat(board, rank_plane, file_plane))
```

Each residual block follows the packet's paired-branch conjunction sketch:

```text
a = Conv3x3_A(h)
b = Conv3x3_B(h)
g = sigmoid(Conv1x1_G(h))
product = a * b
y = Conv1x1(concat(a, b, product, g * a))
h = h + NormActDropout(y)
```

The product feature is an explicit channel group in the residual fusion path; it
is not just a sigmoid gate. The default uses `width: 64`, `depth: 5`,
`branch_width: 32`, `dropout: 0.1`, and `use_coordinate_planes: true`, matching
the research packet defaults. The final head pools the board feature map with
mean and max pooling and classifies the pooled vector with an MLP.

Implemented ablations are selected with the `ablation` config key:
`additive_only`, `gate_only_no_product`, `single_branch_matched`,
`late_product_only`, and `cnn_matched_params`.

Diagnostics expose product branch norm by layer, raw product norm by layer, gate
mean and saturation by layer, branch balance, fusion energy, final feature energy,
and aggregate feature energy.

## Implementation Binding

- Registered model name: `multiplicative_conjunction_convnet`
- Source implementation file: `src/chess_nn_playground/models/multiplicative_conjunction_convnet.py`
- Idea-local wrapper: `ideas/i161_multiplicative_conjunction_convnet/model.py`
