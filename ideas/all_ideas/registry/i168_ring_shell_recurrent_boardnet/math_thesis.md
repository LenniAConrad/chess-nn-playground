# Math Thesis

Ring-Shell Recurrent BoardNet

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `7`.

Working thesis: Important chess context often radiates from anchors: kings,
center squares, edges, and promotion zones. Summarize the board in fixed
rings/shells around these anchors and process the shells with a small
recurrent model.

## Formal model

Let `H \in R^{C \times 8 \times 8}` be the trunk feature map produced by a
compact convolutional encoder of the 18-plane current board. Let
`A = {white_king, black_king, center, white_promotion, black_promotion,
queenside_edge, kingside_edge}` be a fixed family of anchors with positions
`(row_a, col_a)` for `a \in A`. The two king anchors are *dynamic* and use
the soft centroid of the white-king / black-king planes; the remaining
anchors are static.

For each anchor `a` and ring index `r = 0, 1, ..., R-1` define the
Chebyshev shell

```
S_{a, r} = { (h, w) : floor( max(|h - row_a|, |w - col_a|) ) == r }
```

and the per-shell pooled feature

```
f_{a, r} = (1 / |S_{a, r}|) * sum_{(h, w) in S_{a, r}} H[:, h, w]   in R^C.
```

A learned linear projection plus a per-anchor bias produces the radial
sequence `\tilde f_{a, r} = W f_{a, r} + b_a`. A *single shared* GRU
processes every anchor's radial sequence,

```
h_{a, r} = GRU(\tilde f_{a, r}; h_{a, r-1}),
```

so the recurrence integrates information from the anchor outward, shell by
shell. The puzzle logit is then

```
\hat{y} = MLP( LayerNorm( concat_{a in A} h_{a, R-1} ) ),
```

i.e. a small head over the concatenated final hidden states. The
per-anchor / per-ring features and hidden states are exposed as
diagnostics so the radial trail is inspectable without re-running the
model.

## Implementation status

The architecture above is implemented as a bespoke PyTorch model in
`src/chess_nn_playground/models/ring_shell_recurrent_boardnet.py`,
registered under the model name `ring_shell_recurrent_boardnet` and
wrapped by `ideas/all_ideas/registry/i168_ring_shell_recurrent_boardnet/model.py`. It is no
longer a `ResearchPacketProbe` scaffold.
