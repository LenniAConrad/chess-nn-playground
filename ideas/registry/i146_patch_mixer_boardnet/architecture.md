# Architecture

`Patch Mixer BoardNet` is a board-only MLP-Mixer-style classifier over
non-overlapping chess board patches. It is deliberately not a Transformer and
uses no attention; cross-board information moves through token-mixing MLPs and
piece/channel information moves through channel-mixing MLPs.

## Patch Embedding

The model accepts the repository board tensor contract `B x 18 x 8 x 8`.
For the default config it patchifies each board into `2 x 2` square patches:

```text
8 x 8 board -> 4 x 4 patch grid -> 16 tokens
patch_dim = 18 * 2 * 2 = 72
```

`torch.nn.Unfold` extracts the patches. A learned linear projection maps each
flattened patch to `embed_dim` token features.

## Mixer Blocks

Each block follows the packet sketch:

```text
tokens = tokens + token_mlp(norm(tokens).transpose(1, 2)).transpose(1, 2)
tokens = tokens + channel_mlp(norm(tokens))
```

The token MLP mixes the 16 patch positions independently for each embedding
channel. The channel MLP mixes feature channels independently inside each
patch. Stacking these blocks gives a plain non-convolutional baseline for
whole-board patch communication.

## Head

The head concatenates mean and max pooling over patch tokens, then applies a
small MLP to emit one BCE-compatible puzzle logit for the `puzzle_binary`
trainer. Diagnostics include token energy and spread, token/channel mixing
energy, patch occupancy, active patch fraction, patch count, and patch size.

## Ablations

- `patch1_square_mixer`: use `1 x 1` square tokens instead of `2 x 2` patches.
- `patch4_coarse_mixer`: use `4 x 4` coarse patches.
- `no_token_mixing`: remove the token-mixing MLP residual.
- `no_channel_mixing`: remove the channel-mixing MLP residual.
- `cnn_matched_params`: use a plain CNN control head with similar width/depth.

## Implementation Binding

- Registered model name: `patch_mixer_boardnet`
- Source implementation file: `src/chess_nn_playground/models/patch_mixer_boardnet.py`
- Idea-local wrapper: `ideas/registry/i146_patch_mixer_boardnet/model.py`
