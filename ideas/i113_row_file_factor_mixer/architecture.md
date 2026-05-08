# Architecture

`Row-File Factor Mixer` is a bespoke factorized-mixer architecture for
the puzzle_binary contract. There is no convolutional spatial mixing
inside the trunk: spatial structure is processed exclusively by axis
MLPs over ranks and files, with a bilinear rank-file recombination,
and a per-square piece-channel MLP.

## Inputs

- Board tensor only: `(B, 18, 8, 8)` simple_18 contract.
- CRTK / source / engine metadata is reporting-only and never enters
  the model.

## Pipeline

1. **Piece-plane embedding.** A 1x1 convolution lifts the 18 input
   planes to `channels` channels; this performs no spatial mixing.
2. **Mixer blocks (`depth` of them).** Each block, given a residual
   tensor `x : (B, C, 8, 8)`:
   1. Apply LayerNorm over channels and run the **rank MLP** along the
      H axis -- a shared two-layer MLP with hidden width `rank_hidden`,
      shared across files and channels.
   2. In parallel, apply LayerNorm and run the **file MLP** along the
      W axis -- a shared two-layer MLP with hidden width `file_hidden`,
      shared across ranks and channels.
   3. Form the **bilinear rank-file interaction**: elementwise multiply
      the rank-mixed and file-mixed feature maps, LayerNorm over the
      channel axis, and project through a 1x1 conv with `channels`
      output filters.
   4. Add `rank + file + bilinear` to the residual stream.
   5. Apply LayerNorm and run the **piece-channel MLP** over the C axis
      at every square (shared across all 64 squares), and add it as a
      second residual.
3. **Pool and head.** Mean-pool over the spatial axes, LayerNorm the
   pooled feature, then a `Linear(hidden_dim) -> GELU -> Dropout ->
   Linear(1)` head returns the puzzle logit. Per-block rank, file, and
   bilinear energies, plus rank/file summaries and a normalized
   rank-file imbalance score, are returned as diagnostics.

## Tensor Contract

```
input:                      (B, 18, 8, 8)
embedded:                   (B, C, 8, 8)
per-block rank energy:      (B, depth)
per-block file energy:      (B, depth)
per-block bilinear energy:  (B, depth)
rank_summary:               (B, C, 8)        spatial mean over files
file_summary:               (B, C, 8)        spatial mean over ranks
pooled_features:            (B, C)
logits:                     (B,)
rank_file_imbalance:        (B,)
```

## Central Ablations (config switches)

| Ablation        | Config knob                              | Effect                                                                |
|-----------------|------------------------------------------|-----------------------------------------------------------------------|
| `shallow_depth` | `depth: 1`                               | Single mixer block; tests whether one factorized pass suffices.       |
| `wide_axis_mlp` | `rank_hidden`, `file_hidden`             | Increase axis-MLP hidden widths to test capacity along each axis.     |
| `wide_channel`  | `channel_hidden`                         | Increase the piece-channel MLP hidden width independently.            |
| `narrow_trunk`  | `channels: 32`                           | Halve trunk channels to test parameter-budget sensitivity.            |

## Implementation Binding

- Registered model name: `row_file_factor_mixer`
- Source implementation file: `src/chess_nn_playground/models/row_file_factor_mixer.py`
- Idea-local wrapper: `ideas/i113_row_file_factor_mixer/model.py`

The wrapper is a thin adapter over
`build_row_file_factor_mixer_from_config`; it does not touch
`ResearchPacketProbe`. The shared probe wrapper has been removed.
