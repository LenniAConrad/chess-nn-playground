# Architecture

`Tensor-Core Square-Pair Field Network` implements the research packet's dense
`64 x 64` square-pair field directly. The model keeps every ordered square pair in
the forward pass, updates square tokens with dense pair-weight matmuls, and reads
out puzzle-binary evidence from both square-token state and pair-field energies.

## Square Token Projection

Input uses the repository board tensor contract:

```text
x: (B, 18, 8, 8)
```

The board is flattened to 64 square tokens. Each token receives the 18 board
planes plus fixed geometry features:

- rank and file coordinates;
- side-relative rank;
- center distance and edge distance;
- square color parity.

The token projector maps these features to `model_dim`, followed by RMSNorm or
LayerNorm.

## Relation Bank

The implementation precomputes the packet's fixed 18-plane square-pair relation
bank:

```text
R: (18, 64, 64)
```

The relation names are `same_square`, `same_rank`, `same_file`, `same_diag`,
`same_anti_diag`, `same_square_color`, `opposite_square_color`, `knight_offset`,
`king_offset`, `manhattan_distance_1`, `manhattan_distance_2`,
`manhattan_distance_3`, `chebyshev_distance_1`, `chebyshev_distance_2`,
`same_center_ring`, `same_edge_class`, `rank_order_forward`, and
`file_order_forward`.

The `relation_bank_shuffle` ablation uses deterministic shuffled masks that
preserve each relation's density, diagonal count, and symmetry class.

## Pair-Field Blocks

Each block projects square tokens into dense head tensors:

```text
Q, K, V: (B, H, 64, Dh)
```

It forms a pair score without a softmax bottleneck:

```text
pair = Q K^T / sqrt(Dh)
pair += bilinear_pair_rank(Qr, Kr)
pair += relation_mix(R)
```

The pair field is retained for diagnostics. Square messages use the packet's stable
tanh normalization:

```text
weights = tanh(pair) / sqrt(64)
message = weights @ V
```

The message is projected back to `model_dim` and added through a normalized residual
token update, followed by a GELU or SwiGLU token MLP. The implementation loops over
layers only; it does not loop over heads or squares in the forward pass.

## Readout

The square readout concatenates:

- mean token state;
- max token state;
- occupied-square mean;
- king-zone mean.

The pair readout is computed per layer and concatenated across layers. It includes:

- mean absolute pair score, mean squared pair score, and max absolute score;
- row and column pair-energy summaries;
- normalized pair-field entropy proxy;
- per-head energy mean and specialization;
- all 18 relation-conditioned energy summaries;
- occupied-to-occupied, occupied-to-empty, and king-zone pair energies.

The classifier is an MLP over `[square_summary, pair_summary]` and returns one BCE
puzzle logit for the repository's puzzle-binary trainer. Diagnostics include the
architecture-specific pair energies named in the packet.

## Ablation Hooks

Supported `model.ablation` values are `cnn_only_matched`, `no_pair_update`,
`no_pair_readout`, `relation_bank_shuffle`, `softmax_attention_control`,
`low_head_count`, and `pair_energy_only`.

## Implementation Binding

- Registered model name: `tensor_core_square_pair_field_network`.
- Source implementation file: `src/chess_nn_playground/models/trunk/tensor_core_square_pair_field.py`.
- Idea-local wrapper: `ideas/registry/i072_tensor_core_square_pair_field_network/model.py`.
