# Architecture

`Cross-Scale Attention Residual Network` (CSARN) builds 64 fine square tokens
from the current-board `simple_18` tensor and a smaller bank of `K` coarse
tokens by averaging fine tokens over fixed non-overlapping patches of the 8x8
board. It then computes the actual fine-to-fine attention map and a
coarse-anchored prediction of that map factored through the coarse pivots, and
classifies puzzle-likeness from the residual `A_actual - A_predicted`.

## Tensor Contract

```text
input:                       (B, 18, 8, 8)
fine square tokens:          (B, 64, D)
coarse tokens:               (B, K, D)
actual attention A_actual:   (B, 64, 64)
fine -> coarse attention:    (B, 64, K)
coarse -> fine attention:    (B, K, 64)
predicted A_predicted:       (B, 64, 64) = A_fine_to_coarse @ A_coarse_to_fine
residual R = A_act - A_pred: (B, 64, 64)
logits:                      (B,)
```

`K = (8 / coarse_scale)^2`, so `coarse_scale=2` yields 16 coarse tokens (4x4
grid of 2x2 patches) and `coarse_scale=4` yields 4 coarse tokens (2x2 grid of
4x4 patches). The default is `coarse_scale=2`.

## Components

- Square tokenizer: per-square MLP over the 18 input channels concatenated with
  deterministic rank/file/centred coordinates, edge distance, and square
  colour. No engine, search, source, or CRTK metadata participates in the
  forward pass.
- Coarse pooling: averages fine tokens over fixed non-overlapping patches and
  projects them through `LayerNorm`. The patch index buffer is registered as a
  non-persistent buffer so coarse aggregation is deterministic at inference.
- Fine and coarse projections: separate `Q_fine`, `K_fine`, `Q_coarse`,
  `K_coarse` linear projections; no value projection is needed because the
  classifier reads the residual *attention*, not attended values.
- Actual attention: `A_act = softmax(Q_fine K_fine^T / sqrt(D))` over the 64
  square tokens.
- Coarse-anchored prediction: `A_fc = softmax(Q_fine K_coarse^T / sqrt(D))`
  over the `K` coarse pivots, `A_cf = softmax(Q_coarse K_fine^T / sqrt(D))`
  over the 64 fine targets, and `A_pred = A_fc A_cf`. Each row of `A_pred` is
  a convex combination of the `K` coarse-anchored attention rows, so the
  prediction is a rank-`K` factorisation of the fine-to-fine attention through
  the coarse summary.
- Residual map: `R = A_act - A_pred`. Per-row L1 mass measures how much of a
  square's actual attention cannot be explained by any single coarse pivot.
- Residual head: the `(B, 64, 64)` residual is reshaped to `(B, 64, 8, 8)`
  with the source square as channel and the target square as the 8x8 image,
  passed through a small Conv2d-BN-GELU stack, and pooled. The pooled vector
  is concatenated with eight scalar diagnostics (total energy, off-diagonal
  energy, max absolute, self-diagonal mean, asymmetry, Frobenius norm, mean
  row entropy, row entropy variance) and run through a LayerNorm + linear +
  GELU + dropout + linear classifier returning one puzzle logit.

## Output Diagnostics

Forward returns `logits` plus `attention_actual`, `attention_predicted`,
`residual_attention`, `fine_to_coarse_attention`, `coarse_to_fine_attention`,
`residual_features`, and the scalar residual summaries
`residual_total_energy`, `residual_off_diagonal_energy`, `residual_max_abs`,
`residual_self_diagonal_mean`, `residual_asymmetry`, `residual_frobenius`,
`residual_row_entropy_mean`, `residual_row_entropy_variance`, and the per-row
mass `residual_per_source_l1`.

## Implementation Binding

- Registered model name: `cross_scale_attention_residual_network`.
- Source implementation file: `src/chess_nn_playground/models/cross_scale_attention_residual_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i104_cross_scale_attention_residual_network/model.py`.
