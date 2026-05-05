# Architecture

## Architecture Description

The model has four deliberately different branches:

1. grid branch: small CNN over 8x8 planes
2. piece branch: occupied-piece token MLP or tiny transformer
3. relation branch: fixed chess relation pooling over square tokens
4. global branch: material, phase, side-to-move, castling, and coarse occupancy summaries derived from the current board

Each branch emits one evidence logit. The final logit is mean evidence minus disagreement and uncertainty.

## Input Format

First implementation:

```text
batch x 18 x 8 x 8
```

All other views are derived from this tensor and FEN metadata already used by encoders.

## Forward Pass

```text
z_grid = grid_encoder(x)
z_piece = piece_encoder(x)
z_relation = relation_encoder(x)
z_global = global_encoder(x)
e_i, u_i = heads(z_i)
e_bar = mean(e_i)
disagreement = variance(e_i)
uncertainty = mean(softplus(u_i))
residual = small_joint_head(concat(z_i))
logit = e_bar - alpha * disagreement - beta * uncertainty + residual
```

## Tensor Shapes

Suggested first version:

```text
branch_dim: 64
factor_count: 4
factor_logits: batch x 4
output: batch x 1
```

## Output Heads

- main classification logit
- diagnostic factor logits
- diagnostic uncertainty terms

Only the main logit is used for benchmark prediction.

## Parameter Estimate

```text
0.8M to 2M parameters
```

## FLOP Estimate

Comparable to a medium CNN plus small token and relation heads.

