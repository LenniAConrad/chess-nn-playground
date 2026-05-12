# Architecture

`Adapter-Sandwich Residual CNN` realises the source packet's
parameter-efficient-adapter thesis as a bespoke residual CNN for the
repo's `puzzle_binary` task. Each conventional residual block is
sandwiched between two small bottleneck adapters: a *pre*-adapter
applied before the residual block and a *post*-adapter applied after
it. The bulk of capacity stays in the conventional residual blocks;
the adapters add a small amount of structured slack that lets the
network re-route channel mixing locally.

## Implementation Binding

- Registered model name: `adapter_sandwich_residual_cnn`
- Source implementation file: `src/chess_nn_playground/models/adapter_sandwich_residual_cnn.py`
- Idea-local wrapper: `ideas/registry/i154_adapter_sandwich_residual_cnn/model.py`

## Modules

`AdapterSandwichResidualCNN` accepts the project's `(B, 18, 8, 8)`
board tensor only. CRTK / source / engine / verification metadata is
reporting-only and is not consumed.

1. **Stem.** A `3x3` `Conv2d(input_channels -> channels)` followed by
   `BatchNorm2d` and `ReLU` lifts the board planes into the trunk
   channel dimension while preserving the `8 x 8` spatial layout.
2. **Adapter-sandwich stages.** `depth` stages, each:
   1. *Pre-adapter*: a `_BottleneckAdapter` that computes
      `delta = W_up(GELU(W_down(x)))` with `W_down: channels -> adapter_dim`
      and `W_up: adapter_dim -> channels` 1x1 convolutions and adds it
      back to `x` (identity residual).
   2. *Residual block*: two `3x3` `Conv2d` + `BatchNorm2d` layers with
      `ReLU` and `Dropout2d` and an outer identity residual.
   3. *Post-adapter*: a second `_BottleneckAdapter` of the same shape.
3. **Pooling.** `AdaptiveAvgPool2d(1) -> Flatten` produces a single
   `(B, channels)` embedding.
4. **Classifier head.**
   `Linear(channels -> hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim -> 1)`
   emits the one logit required by the puzzle_binary BCE-with-logits
   trainer.

The adapter `W_up` weights and biases are **zero-initialised**, so each
adapter starts as the identity function. At step 0 the whole network is
behaviourally a plain residual CNN; the adapters earn non-zero
contribution only as training proceeds. This is what makes the design a
parameter-efficient capacity knob rather than a competitor backbone.

`adapter_dim` defaults to `max(4, channels // 4)` and can be set
explicitly via the `adapter_dim` config key.

## Loss

The default trainer wires the standard BCE-with-logits on
`output["logits"]`. All adapters and residual blocks share the same
gradient signal through the mean-pooled head; there is no auxiliary
adapter loss.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)`. BCE-compatible log-odds for the one-logit
  puzzle_binary head.
- `logit`, `prob`: aliases of the log-odds and the sigmoid probability.
- `latent`: shape `(B, channels, 8, 8)`, the post-stage feature map.
- `pre_adapter_energy`: shape `(B,)`, the L2-norm sum of the
  pre-adapter `delta` contributions across stages, detached from the
  graph. Reported as a parameter-efficient-capacity diagnostic.
- `post_adapter_energy`: shape `(B,)`, the same for the post-adapters.
- `adapter_energy`: shape `(B,)`, the sum of the two energies.
- `per_stage_pre_adapter_energy`, `per_stage_post_adapter_energy`:
  shape `(B, depth)`, per-stage L2 norms of the adapter deltas.

The energy diagnostics are detached so they are reportable without
biasing the training loss toward minimising or maximising adapter
contribution.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. Engine, verification,
  source, CRTK, principal-variation, mate-score, and best-move metadata
  is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  puzzle_binary BCE-with-logits trainer, plus the diagnostics listed
  above.
- Target mapping: fine labels `0` and `1` map to binary target `0`;
  fine label `2` maps to binary target `1`.
