# Architecture

## Architecture Description

The classifier is a residual stack of chess relation-operator blocks.

Each block takes square embeddings `H` and applies a fixed bank of sparse 64x64 operators. Outputs are mixed by learned low-rank gates and added back through a residual path.

## Input Format

First implementation should use `simple_18`:

```text
batch x 18 x 8 x 8
```

The tensor is reshaped to:

```text
batch x 64 x channels
```

Optional deterministic features:

- square coordinates
- side-to-move plane
- castling/en-passant planes already present if supplied by encoder
- fixed relation masks

## Forward Pass

```text
X = input_projection(board_tensor)
for block in blocks:
    R_k = O_k X for each relation operator k
    g_k = low_rank_context_gate(global_pool(X))
    X = X + MLP(sum_k g_k * R_k)
z_square = mean_pool(X)
z_king = king_zone_pool(X)
z_piece = occupied_square_pool(X)
logit = classifier([z_square, z_king, z_piece])
```

## Tensor Shapes

Suggested first version:

```text
input: batch x 18 x 8 x 8
hidden: batch x 64 x 96
operators: 10-14 sparse 64 x 64 matrices
blocks: 4
output: batch x 1
```

## Output Heads

First benchmark:

```text
puzzle_logit: batch x 1
```

The same trunk should support `num_classes > 1` for future chess classification tasks.

## Parameter Estimate

Rough first configuration:

```text
0.5M to 1.5M parameters
```

Most parameters are channel projections, not operator matrices.

## FLOP Estimate

Sparse operator application is about:

```text
O(batch * edges * channels * blocks)
```

Expected to be cheaper than dense 64-token attention and comparable to a small residual CNN.

