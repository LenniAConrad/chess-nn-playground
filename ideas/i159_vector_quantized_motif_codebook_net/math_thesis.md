# Math Thesis

Vector-Quantized Motif Codebook Net

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.

Batch candidate rank: `5`.

Working thesis: Force local board features to pass through a learned discrete codebook. The classifier reads code usage, spatial code maps, and quantized features. This tests whether a compact inventory of board motifs is useful for puzzle-likeness.

## Formal statement

Let `x in R^{18 x 8 x 8}` be the simple_18 board tensor and let
`f_theta : R^{18 x 8 x 8} -> R^{D x 8 x 8}` be a compact convolutional
encoder producing one motif vector per square. The model maintains a
codebook `C in R^{K x D}` of `K` learned motif prototypes and routes
each square `(i, j)` to its nearest entry:

```
k*(b, i, j) = argmin_{k in [K]} || f_theta(x)[b, :, i, j] - C[k] ||^2
z_q[b, :, i, j] = C[k*(b, i, j)]
```

The straight-through estimator passes encoder gradients through the
quantization step via `z_q_ste = f_theta(x) + stop_grad(z_q - f_theta(x))`,
and the codebook itself is updated with exponential moving averages of
cluster usage and centroid sums. A small head reads three views of the
quantized board: the global mean and max pools of `z_q`, the per-batch
code-usage histogram `p(b) in Delta^{K-1}`, and the average of a learned
spatial code-map embedding `E[k*(b, i, j)]`. The final scalar puzzle
logit is the inner product of the head's last layer with the
concatenated representation. Diagnostics expose `code_usage_entropy`,
`code_perplexity`, `active_codes`, `mean_quantization_distance`, and
the per-sample commitment / codebook MSE losses.

The trained net therefore tests whether a small vocabulary of
square-level motifs (codes shared across positions) is sufficient to
discriminate puzzle from non-puzzle boards under the project's
`puzzle_binary` BCE-with-logits contract.
