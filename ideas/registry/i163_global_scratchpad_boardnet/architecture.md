# Architecture

`Global Scratchpad BoardNet` is a board-only classifier for the `puzzle_binary`
task. It accepts the repo's simple 18-plane current-board tensor with shape
`(B, 18, 8, 8)` and returns one puzzle logit per position.

The board encoder is a compact coordinate-aware CNN stem:

```text
h0 = CNNStem(concat(board, rank_plane, file_plane))
```

The model maintains a small fixed set of global memory slots. Initial memory is
the sum of learned slot vectors and a board-conditioned projection:

```text
m0 = learned_memory + MLP(global_pool(h0))
```

For each scratchpad step, the board sends fixed pooled summaries into every
memory slot. The summary uses global board pooling and coordinate-weighted board
pooling, not square-to-memory attention:

```text
summary_t = pool([h_t, h_t * rank, h_t * file])
m_{t+1} = GRUCell(summary_t, m_t)
```

The updated memory broadcasts global context back to every square through FiLM
modulation. A residual convolutional update keeps the recurrent scratchpad
stable:

```text
film_t = MLP(mean/max_pool_slots(m_{t+1})) -> gamma_t, beta_t
h_{t+1} = h_t + 0.25 * (ConvBlock(gamma_t * h_t + beta_t) - h_t)
```

The classifier reads pooled final board features and pooled final memory slots:

```text
z = concat(mean/max/std_pool(h_T), mean_pool(m_T), max_pool(m_T))
logits = MLP(z)
```

Implemented ablations are `no_scratchpad`, `one_step`, `no_broadcast`,
`random_memory`, and `single_slot`.

Diagnostics include memory slot norms by step, memory update norms by step,
board activation change after each broadcast, final memory slot similarity,
board feature energy, and scratchpad step/slot counts.

## Implementation Binding

- Registered model name: `global_scratchpad_boardnet`
- Source implementation file: `src/chess_nn_playground/models/trunk/global_scratchpad_boardnet.py`
- Idea-local wrapper: `ideas/registry/i163_global_scratchpad_boardnet/model.py`
