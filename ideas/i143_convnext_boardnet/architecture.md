# Architecture

`ConvNeXt BoardNet` is a small ConvNeXt-style classifier specialized to
the `8 x 8` chess board. Each block follows the canonical ConvNeXt
recipe — depthwise spatial mixing, channel-wise LayerNorm, an inverted
channel MLP, and a learned residual scale (LayerScale) — but at chess
board resolution there is no spatial downsampling: every block keeps the
full `8 x 8` grid so all 64 squares retain their identity through the
trunk.

## Pipeline

- Input: board tensor `(B, input_channels, 8, 8)` (defaults to the
  `simple_18` encoding). CRTK / source / engine metadata is
  reporting-only and never reaches the model.
- Coordinate planes: a deterministic 4-channel `BoardCoordinatePlanes`
  module concatenates rank, file, center-distance, and
  square-color planes onto the input, giving the trunk explicit access
  to absolute board geometry.
- Stem: a single `3 x 3` conv lifts `input_channels + 4` to `channels`,
  followed by a channel-wise `LayerNorm2d` and `GELU`.
- Trunk: `depth` `ConvNeXtBoardBlock`s, each implementing
  `x + gamma * MLP(LayerNorm(DepthwiseConv(x)))` where:
  - `DepthwiseConv` is a `kernel_size x kernel_size` conv with
    `groups = channels` (`kernel_size` defaults to 3, must be odd),
  - `LayerNorm` is applied per spatial location across the channel axis,
  - the channel MLP is `Linear(channels, mlp_hidden_dim) -> GELU ->
    optional Dropout -> Linear(mlp_hidden_dim, channels)` and is
    *inverted* in the ConvNeXt sense (`mlp_hidden_dim > channels`,
    enforced at construction time),
  - `gamma` is a learned per-channel `LayerScale` parameter initialized
    to `1.0e-3`.
- Final norm: a `LayerNorm2d` over the trunk output.
- Pooling head (`ConvNeXtBoardPoolingHead`): combines four pooled
  views of the `(B, channels, 8, 8)` feature map:
  - global mean pool,
  - global max pool,
  - per-channel spatial std (population, `unbiased=False`),
  - learned attention pool — `softmax` over a `1 x 1` conv attention map
    produces a per-square distribution that weights the spatial
    features.
  The four `channels`-dim pools are concatenated to a `4 * channels`
  vector and passed through `LayerNorm -> Linear(4*channels, hidden)
  -> GELU -> Dropout -> Linear(hidden, hidden/2) -> GELU -> Dropout
  -> Linear(hidden/2, num_classes)` producing one puzzle logit.
- Output `dict`:
  - `logits`: shape `(B,)` for `num_classes == 1`, `(B, num_classes)`
    otherwise.
  - Diagnostics from the pooling head:
    `pool_attention_entropy`, `pool_attention_peak`,
    `spatial_contrast`, `feature_std`.
  - Trunk-level diagnostics: `convnext_feature_energy` (mean squared
    feature activation), `coordinate_response` (squared response of the
    stem to the coordinate planes only), and a `piece_density` proxy
    pooled over the first piece planes.

## Why It Is Distinct

- Not a generic CNN baseline: the trunk runs ConvNeXt blocks with
  depthwise spatial mixing and inverted channel MLPs at constant
  `8 x 8` resolution; every per-square representation is
  channel-mixed by the inverted MLP rather than further spatial conv.
- Not a `ResearchPacketProbe`: the head consumes ConvNeXt features
  directly, including a learned attention pool, std pool, and
  coordinate-aware stem. There is no proposal-profile branch and no
  packet-keyword diagnostic.
- LayerScale + identity residuals follow the ConvNeXt residual scaling
  recipe so depth can be increased without retuning.

## Implementation Binding

- Registered model name: `convnext_boardnet` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/convnext_boardnet.py`
  (`ConvNeXtBoardNet` and
  `build_convnext_boardnet_from_config`).
- Idea-local wrapper:
  `ideas/i143_convnext_boardnet/model.py` calls
  `build_convnext_boardnet_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this idea.
