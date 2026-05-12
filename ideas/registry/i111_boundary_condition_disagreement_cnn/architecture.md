# Architecture

`Boundary-Condition Disagreement CNN` is a board-only puzzle_binary classifier
that runs a *shared-weight* convolutional trunk under several boundary
conditions and reads the disagreement between the per-mode feature streams.
There is no proposal-profile diagnostics, no mechanism-family embedding and
no shared `ResearchPacketProbe` involvement: the head input is exactly the
multi-boundary disagreement decomposition prescribed by `math_thesis.md`.

## Pipeline

1. The trunk is a stack of `depth` shared-weight convolution blocks. Each
   block defines one set of weights `W` and runs one explicit
   ``F.pad`` -> ``F.conv2d(padding=0)`` -> ``GroupNorm`` -> ``GELU`` ->
   ``Dropout2d`` pass per boundary mode in `boundary_modes`. The boundary
   mode is realised by `F.pad(mode=...)` with the supported choices
   `zeros`, `reflect`, `replicate`, `circular`. Weights, biases and
   normalisation are shared across modes; only the ghost frame around the
   8x8 grid changes.
2. After `depth` blocks each boundary mode `m` has produced a feature map
   `F_m : (B, channels, 8, 8)`. Stacking them gives
   `boundary_features : (M, B, channels, 8, 8)` where `M = len(boundary_modes)`.
3. The disagreement map is the per-position variance across boundary modes,
   `disagreement_map = boundary_features.var(dim=0, unbiased=False)` of
   shape `(B, channels, 8, 8)`. A diagnostic
   `pairwise_disagreement_energy : (B, M, M)` matrix of mean squared
   pairwise differences between modes is also reported.
4. The classifier head receives, in this order:
   * **per-mode pooled features** -- for each boundary mode `m`, the
     channel-wise mean and max pool of `F_m` over the 8x8 grid, each
     `(B, channels)`, concatenated to `(B, 2*channels)` per mode;
   * **disagreement-pooled features** -- the channel-wise mean and max
     pool of `disagreement_map` over the 8x8 grid, each `(B, channels)`,
     concatenated to `(B, 2*channels)`.
5. The head is `LayerNorm -> Linear(head_input -> hidden_dim) -> GELU
   -> Dropout -> Linear(hidden_dim -> 1)` and emits a single puzzle logit
   `logits : (B,)`. All intermediate signals are exposed alongside the
   logit so ablations and reports can read them without a second forward
   pass.

## Tensor Contract

```text
input:                          (B, 18, 8, 8)
boundary_features:              (M, B, channels, 8, 8)
disagreement_map:               (B, channels, 8, 8)
disagreement_mean:              (B, channels)
disagreement_max:               (B, channels)
disagreement_energy:            (B, 2*channels)
pairwise_disagreement_energy:   (B, M, M)
per_mode_mean:                  (B, M, channels)
per_mode_max:                   (B, M, channels)
per_mode_pooled:                (B, M, 2*channels)
logits:                         (B,)
```

The head input dimensionality is `M * 2 * channels + 2 * channels`.

## Why this is not a shared probe

The model has no proposal-profile diagnostics, no mechanism-family
embedding, and no `ResearchPacketProbe` code. The signal that reaches the
head is exactly the multi-boundary disagreement decomposition prescribed
by `math_thesis.md`: per-mode pooled feature vectors plus per-position
variance across boundary modes. Ablations on `boundary_modes` (e.g. drop
`circular` to revert to a classical `zeros + reflect + replicate`
ensemble), `depth`, `channels`, and `hidden_dim` map directly to the
central design knobs in the source packet, and ablations that hide
individual components (drop the disagreement summary, or drop a single
boundary mode) are well-defined operations on this code path.

## Implementation Binding

- Registered model name: `boundary_condition_disagreement_cnn`.
- Source implementation file:
  `src/chess_nn_playground/models/trunk/boundary_condition_disagreement_cnn.py`.
- Idea-local wrapper:
  `ideas/registry/i111_boundary_condition_disagreement_cnn/model.py` (a thin
  `build_model_from_config` over
  `build_boundary_condition_disagreement_cnn_from_config`; no
  `ResearchPacketProbe` is involved).
