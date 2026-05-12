# Architecture

`Agreement-Variance Head Net` realises the source packet's
shared-trunk-with-many-cheap-heads classifier as a bespoke PyTorch
model for the repo's `puzzle_binary` task. One convolutional trunk
encodes the board, and `num_heads` independent cheap MLP heads each
emit a single logit. The reported classification logit is the *mean*
of the per-head logits; the *variance* across heads is exposed as an
uncertainty diagnostic and is intentionally not part of the gradient.

This is a lightweight alternative to a full ensemble: one trunk forward
pass produces `num_heads` predictions instead of `num_heads` independent
models.

## Implementation Binding

- Registered model name: `agreement_variance_head_net`
- Source implementation file: `src/chess_nn_playground/models/agreement_variance_head_net.py`
- Idea-local wrapper: `ideas/registry/i153_agreement_variance_head_net/model.py`

## Modules

`AgreementVarianceHeadNet` accepts the project's `(B, 18, 8, 8)` board
tensor only. CRTK / source / engine / verification metadata is
reporting-only and is not consumed.

1. **Stem.** A `3x3` `Conv2d(input_channels -> channels)` followed by
   `BatchNorm2d` and `ReLU` lifts the board planes into the trunk
   channel dimension while preserving the `8 x 8` spatial layout.
2. **Trunk.** `depth` `_ResidualBlock` units (two `3x3` `Conv2d` layers
   with `BatchNorm2d`, `ReLU`, and `Dropout2d`) refine the latent
   feature map.
3. **Pooled embedding.** `AdaptiveAvgPool2d(1) -> Flatten` produces a
   single `(B, channels)` embedding shared by all heads.
4. **Heads.** `num_heads` independently initialised
   `Linear(channels, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)`
   modules. Each head is small and cheap so the total cost is close to
   one trunk plus a few MLP heads, materially less than `num_heads`
   independent CNNs.
5. **Aggregation.** Per-head logits are stacked into a `(B, num_heads)`
   tensor. The mean across heads is the classification logit. The
   variance (and its square root, the disagreement) is computed under
   `torch.no_grad()` so the loss does not implicitly minimise it.

## Loss

The default trainer wires the standard BCE-with-logits on
`output["logits"]`. Each head only receives gradients through the
`mean(logit)` aggregation, so all heads are trained on the same target
without any explicit agreement penalty. Independent random init of
each head's `Linear` weights keeps the heads from collapsing to the
same function.

## Diagnostics

`forward` returns a dict containing:

- `logits`: shape `(B,)`. BCE-compatible mean log-odds for the one-logit
  `puzzle_binary` head.
- `logit`, `prob`: aliases of the mean log-odds and the corresponding
  sigmoid probability.
- `per_head_logits`: shape `(B, num_heads)`, the raw per-head logits.
- `per_head_probs`: shape `(B, num_heads)`, the sigmoid of the per-head
  logits (detached).
- `head_variance`: shape `(B,)`, variance of the per-head logits
  (detached). Reported as the agreement diagnostic.
- `head_disagreement`: shape `(B,)`, square root of `head_variance`.
- `prob_variance`: shape `(B,)`, variance of the per-head probabilities.
- `latent`: shape `(B, channels, 8, 8)`, the shared trunk feature map.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. Engine, verification, source,
  CRTK, principal-variation, mate-score, and best-move metadata is
  reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit
  `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed
  above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine
  label `2` maps to binary target `1`.
