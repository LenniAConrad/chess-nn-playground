# Math Thesis

Spatial FiLM Coordinate Net

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.

Batch candidate rank: `4`.

Working thesis: Appending coordinate planes may be too weak. Instead, generate
per-square affine modulation parameters from deterministic coordinate features
and side-relative coordinates, then modulate CNN features at multiple depths.

The deterministic coordinate vector at each square `s` is

```text
c_s = [rank, file, center_distance, edge_distance,
       side_relative_rank, square_color]
```

with `rank, file, side_relative_rank` in `[-1, 1]`, `center_distance` the
Chebyshev distance to the board centre, `edge_distance = 1 - center_distance`,
and `square_color` in `{-1, +1}`. A small per-layer MLP turns `c_s` into
modulation parameters

```text
gamma_l(s), beta_l(s) = MLP_l(c_s)
```

bounded by `gamma_l = 1 + 0.25 * tanh(raw_gamma_l)` and
`beta_l = 0.25 * tanh(raw_beta_l)`. At every conv layer, after the convolution
and norm/activation block, features are modulated by

```text
h_l = gamma_l(s) * ConvBlock(h_{l-1}) + beta_l(s).
```

Since `c_s` is a deterministic function of the board square, the FiLM
generators effectively learn a fixed coordinate-indexed look-up, but the
gating is applied at every depth and is conditioned on a richer feature set
than plain rank/file planes. Bounded modulation keeps the trunk close to a
plain CNN at initialization, so training only adds spatial modulation as
needed.
