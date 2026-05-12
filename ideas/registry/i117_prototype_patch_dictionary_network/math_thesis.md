# Math Thesis

Prototype Patch Dictionary Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `5`.

Working thesis: Puzzle-like positions may contain local motifs, but a
standard CNN may hide them in distributed filters. A learned patch
dictionary can expose motif assignments, reconstruction residuals, and
prototype activation histograms.

## Model

Let `p_{b, s} in R^D` be the patch embedding of square `s` on board
`b`, produced by a small convolutional encoder over the simple_18
board tensor. The model holds a learned dictionary
`D = [d_1, ..., d_K] in R^{K x D}` of motif prototypes.

For every square the model computes a soft assignment by
cosine-similarity softmax over the dictionary:

```
similarity_{b, s, k} = <normalize(p_{b, s}), normalize(d_k)>
alpha_{b, s, k}      = softmax_k( similarity_{b, s, k} / tau )
```

with a learned positive temperature `tau = exp(log_tau)`. The patch is
reconstructed by the convex combination
`p_hat_{b, s} = sum_k alpha_{b, s, k} * d_k`, which yields the residual
`r_{b, s} = p_{b, s} - p_hat_{b, s}`.

The puzzle classifier reads three readouts directly:

1. The motif assignment map `alpha` (and its `argmax_k`, the top-1
   motif id per square).
2. The reconstruction residual `r` and its per-square energy
   `||r_{b, s}||^2`.
3. The prototype activation histogram
   `h_{b, k} = sum_s alpha_{b, s, k} / 64`,
   which is a probability mass over the `K` prototypes.

The head consumes `concat(h, mean_s |r|, mean_s p, mean_s p_hat)` and
returns one puzzle logit.

## Why this is materially distinct from the shared probe / vanilla CNN

The shared `ResearchPacketProbe` scaffold has no dictionary, no soft
assignment, no reconstruction, and no prototype histogram. A vanilla
CNN with `K` channels can produce a `(B, K, 8, 8)` map, but that map is
not coupled to a fixed set of vectors that *also* reconstruct the
patches. The same dictionary directions `d_k` are used in both the
assignment softmax and the reconstruction sum, which is what makes the
assignment a "motif id" and the residual a "reconstruction failure"
rather than an arbitrary feature map.
