# Math Thesis

Early-Exit Cascade BoardNet.

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2210_friday_shanghai_architecture_batch_9.md`.

Batch candidate rank: `2`.

The puzzle decision is modeled as the expectation of a per-exit puzzle
probability under a forward-halting distribution along a depth-staged
trunk:

\[
\Pr(y = 1 \mid x)
= \sum_{k=0}^{K-1} w_k(x)\, \sigma(z_k(x)),
\quad
w_k(x) = h_k(x) \prod_{j<k} (1 - h_j(x))
\quad \text{for } k < K-1,
\quad
w_{K-1}(x) = \prod_{j=0}^{K-2} (1 - h_j(x)),
\]

with `h_k = sigma(s_k / tau)` the learned halting probability at exit
`k < K - 1`, `z_k` the per-exit puzzle logit, and `tau` a temperature.
The cascade weights satisfy `sum_k w_k = 1` exactly. The model emits

\[
\text{logit}(x) = \log \frac{p(x)}{1 - p(x)},
\quad p(x) = \sum_k w_k(x)\, \sigma(z_k(x)),
\]

so a single BCE-with-logits loss on this quantity differentiates with
respect to every `z_k` and every `s_k`, and the cascade trains end-to-end
without trainer modifications. An auxiliary per-exit BCE
`(1 / K) sum_k \mathrm{BCE}(z_k, y)` is exposed for ablation use.

The thesis is that the puzzle target is well predicted by a small number
of stage-wise board exits whose halting probability is itself learned: easy
positions can be decided by an early exit (small `expected_exit_index`),
while ambiguous near-puzzles are deferred to deeper exits. Falsification
proceeds by collapsing the gates: if the optimum sets `h_0 ~ 0` for every
example so that all weight ends up on the deepest exit, the cascade has
not earned the early-exit structure relative to a single deep classifier.
The model exposes `expected_exit_index`, the per-exit logit stack, and the
per-exit weight stack so this collapse is directly observable in metrics.
