# Math Thesis

Neural Board Cellular Automaton

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.

Batch candidate rank: `3`.

Working thesis: Some board patterns may be recognized by repeated local
relaxation. A neural cellular automaton applies the same local update
rule for several steps and classifies from the evolving board state and
update energy.

## Model

Let `h_t \in R^{C \times 8 \times 8}` be the cell-state board after
step `t`. The model uses one shared local rule `f` (a small 3x3
convnet) and runs the residual relaxation

```
h_0     = W_embed * x
h_{t+1} = h_t + step_size * f(h_t),     t = 0, 1, ..., T-1
```

with the same `f` at every step (tied weights). The puzzle logit is

```
y_hat = head( pool(h_T) || E(h_0..h_T) )
```

where `pool` is spatial mean-pool and `E(.)` is a fixed-size summary of
the per-step update energies `||h_{t+1} - h_t||^2 / N` and state
energies `||h_t||^2 / N` (mean, last, sum statistics over `t`).

The relaxation interpretation is that easy positions converge to a
stable cell state quickly so update energy decays toward zero, while
positions with tactical content sustain non-trivial updates. The
classifier therefore reads both *what* the relaxed state encodes and
*how much it had to relax* to get there.

## Initialization

`f` is initialized so its final 1x1 output is zero. This makes the
untrained dynamics `h_{t+1} = h_t` the identity fixed point, which is
stable for any `T`. Training learns useful relaxation away from that
fixed point. `step_size = sigmoid(theta) * max_step_size` keeps the
per-step magnitude bounded throughout training.
