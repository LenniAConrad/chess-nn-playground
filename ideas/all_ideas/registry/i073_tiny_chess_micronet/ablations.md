# Ablations

Set `model.ablation` to one of:

- `counts_only_mlp`: zero all spatial descriptors and classify from material/state counts.
- `ordinary_tiny_cnn_matched`: replace the squeeze plus line blocks with a tiny local CNN and keep only global pools plus material counts.
- `flat_head_same_params`: use a learned flat hidden-map descriptor path instead of the deterministic sketch bank.
- `no_line_sketch`: remove rank/file/diagonal/anti-diagonal sketch descriptors.
- `random_line_basis`: replace chess-shaped line bases with deterministic random bases of the same dimensions.
- `no_king_zone`: remove own/opponent king-zone pools.
- `no_depthwise_local`: keep the squeeze and sketch bank but remove micro depthwise-line blocks.

The central falsifiers are `counts_only_mlp`, `ordinary_tiny_cnn_matched`,
`no_line_sketch`, and `random_line_basis`.
