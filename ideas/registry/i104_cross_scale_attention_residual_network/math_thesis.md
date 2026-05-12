# Math Thesis

Cross-Scale Attention Residual Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2056_friday_shanghai_attention_inspired_batch.md`.

Batch candidate rank: `3`.

Working thesis: Puzzle-like evidence may appear when fine-square attention
cannot be predicted from coarse board context. The model computes the actual
fine-to-fine attention `A_act = softmax(Q_F K_F^T / sqrt(D))` over the 64
square tokens, and reconstructs an expected fine-to-fine attention through
`K` coarse pivots as `A_pred = A_fc A_cf`, where
`A_fc = softmax(Q_F K_C^T / sqrt(D))` and
`A_cf = softmax(Q_C K_F^T / sqrt(D))`. Each row of `A_pred` is a convex
combination of `K` coarse-anchored rows, so the prediction is a rank-`K`
factorisation of the fine-to-fine attention through the coarse summary. The
classifier reads the residual `R = A_act - A_pred`: per-row L1 mass measures
how much of a square's actual attention cannot be explained by any single
coarse pivot, and the puzzle logit is decided from those residual statistics
plus a small Conv2d head over `R` reshaped as `(B, 64, 8, 8)` with the source
square as channel and the target square as the 8x8 image.
