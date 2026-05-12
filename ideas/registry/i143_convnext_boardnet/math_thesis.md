# Math Thesis

ConvNeXt BoardNet

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `1`.

Working thesis: Use a small ConvNeXt-style architecture adapted to `8 x 8` chess boards: depthwise spatial mixing, inverted channel MLPs, residual scaling, coordinate planes, and a strong global pooling head.

## Formal Architecture

Let `x \in R^{B x C_in x 8 x 8}` be the board tensor. We construct a
deterministic coordinate-plane field `P \in R^{4 x 8 x 8}` of rank,
file, center-distance, and square-color values and form the stem input
`x_0 = [x ; P_repeat] \in R^{B x (C_in + 4) x 8 x 8}`. The stem is a
single `3 x 3` convolution with channel-wise `LayerNorm2d` and `GELU`
producing `h_0 \in R^{B x C x 8 x 8}` with `C = channels`.

A ConvNeXt board block applies, with depthwise convolution `DW` of
kernel `k`, channel-wise `LayerNorm` (LN), inverted MLP
`Phi: R^C -> R^{C_h} -> R^C` (`C_h = mlp_hidden_dim > C`), and
LayerScale parameter `gamma \in R^C`:

    h_{l+1} = h_l + gamma \odot Phi(LN(DW_k(h_l)))

The transformation factorizes spatial mixing (DW) from channel mixing
(`Phi`) so the inverted MLP `Phi` does the channel-rank role and the
depthwise conv does the spatial-rank role. Setting `gamma` small at
init leaves the residual stream dominant and allows stable depth
scaling.

After `depth` blocks we apply a final `LayerNorm2d` to obtain
`h_L \in R^{B x C x 8 x 8}`.

The pooling head produces four `C`-dim summaries:

- `mu = (1/64) sum_{ij} h_L[:, :, i, j]` (mean pool),
- `M = max_{ij} h_L[:, :, i, j]` (max pool),
- `s_pop = sqrt((1/64) sum_{ij} (h_L - mu)^2)` (population std),
- `a = sum_{ij} alpha_{ij} h_L[:, :, i, j]`,
  with `alpha = softmax_{ij}(W_a h_L)` from a `1 x 1` attention conv.

We concatenate `[mu; M; s_pop; a] \in R^{4C}` and pass through a
`LayerNorm + 3-layer MLP` to a single logit. The squeezed attention
distribution `alpha` also yields `pool_attention_entropy` and
`pool_attention_peak` diagnostics.

## Loss

Binary puzzle classification with `BCEWithLogitsLoss`, with class
weighting set by the trainer (`class_weighting: balanced` in
`config.yaml`).
