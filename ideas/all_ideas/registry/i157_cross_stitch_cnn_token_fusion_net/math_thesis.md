# Math Thesis

Cross-Stitch CNN-Token Fusion Net

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2213_friday_shanghai_architecture_batch_10.md`.

Batch candidate rank: `3`.

Working thesis: Late fusion between a CNN branch and a piece-token
branch may be too weak. A cross-stitch network can let the branches
exchange information at multiple depths through learned linear mixing,
while still keeping the model practical.

## Cross-Stitch Mixing

Let `h_board^t in R^{B x C x 8 x 8}` be the board branch and
`h_token^t in R^{B x P x C}` be the token branch at stage `t`. At
each stage we summarise the branches by

```
b^t = pool(h_board^t)   in R^{B x C}
p^t = masked_mean(h_token^t)   in R^{B x C}
```

and apply a learned ``2x2`` (or per-group ``G x 2 x 2``) cross-stitch
matrix ``A^t``:

```
[b'^t_g]   [a^t_g  c^t_g] [b^t_g]
[p'^t_g] = [d^t_g  e^t_g] [p^t_g]
```

where the channel-``C`` summaries are split into ``G`` equal groups and
``A^t in R^{G x 2 x 2}`` is initialised to the identity. The mixed
summaries are injected back into each branch through learned linear
adapters:

```
h_board^{t+1} = h_board^t + board_adapter^t(b'^t)
h_token^{t+1} = h_token^t + token_adapter^t(p'^t)   (masked)
```

The final pooled summaries plus the per-stage cross-stitch off-diagonal
energy drive a small MLP head producing the puzzle logit.

Identity initialisation guarantees that the model recovers the parent
late-fusion behaviour at the start of training; learned off-diagonal
entries quantify the amount of board-to-token and token-to-board
transfer at each depth. The architecture deliberately keeps the
fusion linear (a ``2x2`` matrix per group) rather than attentional, so
the diagnostics directly read out as branch-mixing coefficients.

## Bespoke Implementation

This folder is a bespoke `CrossStitchCNNTokenFusionNet` model in
`src/chess_nn_playground/models/cross_stitch_cnn_token_fusion_net.py`.
It is not a `ResearchPacketProbe` scaffold.
