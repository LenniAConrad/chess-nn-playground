# Architecture

`Hypercolumn Square Readout CNN` is a board-only convolutional classifier for the
`puzzle_binary` task. It accepts the repo's simple 18-plane current-board tensor
with shape `(B, 18, 8, 8)` and returns one puzzle logit per position.

The trunk is a compact residual CNN that preserves the board grid at every depth:

```text
h1, h2, h3, h4 = trunk_layers(x)
```

Each saved layer is projected to a common per-square width with an independent
`1x1` convolution:

```text
p_t = Conv1x1(h_t -> hyper_width)
```

The projected layer maps are concatenated at every square to form a hypercolumn:

```text
H_square = concat(p1, p2, p3, p4)  # (B, 4 * hyper_width, 8, 8)
```

A square evidence head reads each hypercolumn, emits local evidence features, and
then produces a two-channel square-logit map:

```text
e = Conv1x1(H_square -> evidence_width)
e = Conv3x3(e -> evidence_width)
square_logits = Conv1x1(e -> 2)
```

The final classifier aggregates dense and sparse square evidence:

```text
z = concat(mean_pool(e), max_pool(e), topk_pool(square_logits))
logits = MLP(z)
```

The implementation exposes diagnostics for the markdown's intended inspection
surface: square evidence maps, square logits, layer projection energy, early/late
projection dominance, and top evidence square indices. Central ablations are
available through the `ablation` config key: `last_layer_only`, `no_square_logits`,
`mean_pool_only`, `cnn_head_matched`, and `random_layer_order`.

## Implementation Binding

- Registered model name: `hypercolumn_square_readout_cnn`
- Source implementation file: `src/chess_nn_playground/models/hypercolumn_square_readout_cnn.py`
- Idea-local wrapper: `ideas/i160_hypercolumn_square_readout_cnn/model.py`
