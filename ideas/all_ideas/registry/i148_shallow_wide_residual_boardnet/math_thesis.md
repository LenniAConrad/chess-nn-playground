# Math Thesis

Shallow Wide Residual BoardNet

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `6`.

Working thesis: On an `8 x 8` board, depth may be less useful than width
and a good head.  A shallow wide residual CNN can test whether the
benchmark wants broad feature extraction rather than long convolutional
stacks.

## Architecture in symbols

Let `x in R^{B x 18 x 8 x 8}` be the simple_18 board tensor and
`c in R^{B x 2 x 8 x 8}` be the coordinate planes carrying linear
rank and file in `[-1, 1]`.  Define the stem

```text
h_0 = ReLU(BN(Conv3x3(concat(x, c) -> W)))
```

at width `W` (default 96).  Each residual block is

```text
r_l   = SE_l(BN(Conv3x3(ReLU(BN(Conv3x3(h_{l - 1}))))))
h_l   = ReLU(h_{l - 1} + r_l)
SE_l  = sigmoid(W2 ReLU(W1 GAP(.)))  channel-attention gate
```

The pooled head reads three first/second-order statistics

```text
mu     = GAP(h_L)
M      = GMaxP(h_L)
sigma  = sqrt(GAP((h_L - mu) ^ 2))
```

and the trunk logit is

```text
y_trunk = Linear(ReLU(Linear(LN(concat(mu, M, sigma)))))
```

When `use_count_head` is on, an explicit count vector
`m = sum_{rank, file} x` is fed through a small MLP to produce
`y_count`, and the puzzle logit is `y = y_trunk + y_count`.

## Why this answers the thesis

If a wide but shallow `(W = 96 or 128, L in {2, 3})` trunk plus the
pooled head matches or beats the deeper `residual_cnn` baseline, the
benchmark is dominated by broad feature extraction over a small
spatial domain rather than long convolutional reasoning chains.  The
SE gate and the optional count head add ablate-able pieces that let us
attribute any gain to channel attention or to material short-circuits
rather than to width alone.
