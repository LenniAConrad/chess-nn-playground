# Architecture

`Shallow Wide Residual BoardNet` is a bespoke implementation of idea
`i148`.  The model deliberately trades depth for width on the `8 x 8`
board: a wide stem feeds a small stack of squeeze-excite residual
blocks, then a strong pooled head reads mean / max / std statistics
and an optional per-side material count vector.

## Pipeline

- Input: board tensor `(B, 18, 8, 8)`.  CRTK / source metadata is
  reporting-only and never used as model input.
- Optional coordinate planes: two extra channels carrying linear
  rank and file coordinates in `[-1, 1]` are concatenated to the
  input so the network has explicit absolute square information.
- Wide stem: `Conv3x3(input + coord -> width) -> BN -> ReLU`.
- Body: `depth` residual blocks at constant width.  Each block is

  ```text
  h -> Conv3x3 -> BN -> ReLU -> Dropout2d -> Conv3x3 -> BN -> SE-gate
       |                                                          |
       +------------------ skip ----------------------------------+
                                  |
                                  +-> ReLU
  ```

  The squeeze-excite gate computes per-channel attention via
  `GAP -> Linear -> ReLU -> Linear -> sigmoid` and modulates the
  block's residual stream.

- Pooled head: mean / max / std pooling are concatenated along the
  channel axis to a `3 * width` feature vector.  This vector is fed
  through `LayerNorm -> Linear -> ReLU -> Dropout -> Linear` to a
  single trunk logit.

- Optional count head: a side path consumes the per-channel sums of
  the raw simple_18 input through `Linear -> ReLU -> Linear` to a
  scalar count logit.  When enabled it is added to the trunk logit
  so the head can short-circuit decisions that depend only on
  material / role counts.

## Capacity defaults

The packet calls for `width = 96 or 128` and `depth = 2 or 3`.  The
default config (`channels = 64`, `depth = 2`, `hidden_dim = 96`) sits
at the bottom of that range; the model also accepts the packet-style
`width` config key as an alias for `channels`.

## Diagnostics

The forward returns a dict with the puzzle logit and a fixed set of
diagnostic tensors that confirm the wide trunk, SE gate and pooled
head are all active:

- `swrb_pool_mean_norm` - L2 norm of the mean-pooled trunk features.
- `swrb_pool_max_max` - max value of the max-pooled trunk features.
- `swrb_pool_std_norm` - L2 norm of the std-pooled trunk features.
- `swrb_se_gate_mean` - average SE-gate activation per sample,
  averaged over blocks and channels.
- `swrb_residual_energy` - mean squared body residual contribution
  at the final block.
- `swrb_count_head_logit` - logit contribution of the side count
  head (zero when `use_count_head` is false).

## Why this is distinct

- Not the deeper `residual_cnn` baseline: this trunk is wider but
  much shallower (2-3 blocks vs. 6 by default), every block ends in
  a squeeze-excite gate (the baseline has none), and the head pools
  three statistics instead of a single mean pool.
- Not a research-packet probe: there is no shared mechanism profile,
  no proposal diagnostics, and no reliance on
  `ResearchPacketProbe` or `build_research_packet_probe_from_config`.
- Not a CNN with positional embeddings only: the optional coordinate
  planes are concatenated at the input rather than added inside an
  attention layer, and the SE gate operates purely on channels.

## Implementation Binding

- Registered model name: `shallow_wide_residual_boardnet`
  (registered in `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/shallow_wide_residual_boardnet.py`
  (`ShallowWideResidualBoardNet` and
  `build_shallow_wide_residual_boardnet_from_config`).
- Idea-local wrapper:
  `ideas/registry/i148_shallow_wide_residual_boardnet/model.py` calls
  `build_shallow_wide_residual_boardnet_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by
  this idea.
