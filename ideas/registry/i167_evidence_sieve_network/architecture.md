# Architecture

`Evidence Sieve Network` is a board-only classifier for the `puzzle_binary`
task. It accepts the repo's simple 18-plane current-board tensor with shape
`(B, 18, 8, 8)` and returns one puzzle logit per position together with the
per-stage sieve trail.

## Trunk

The trunk is compact: a `Conv3x3 -> BatchNorm -> GELU` stem followed by
`depth` `ConvBlock`s (`Conv3x3 -> BatchNorm -> GELU -> Dropout`) of width
`channels`. Its output is the initial feature map
`H_0 \in R^{channels \times 8 \times 8}`.

## Sieve stack

`T = num_sieves` evidence sieve stages refine the feature map by filtering
channels and squares. For `t = 1..T`,

- Channel mask: `c_t = sigmoid( MLP([GAP(H_{t-1}); GMP(H_{t-1})]) )` with
  hidden width `channel_gate_hidden`. Concatenating the global average and
  global max pools lets the gate see both ``how much`` and ``the strongest``
  activation per channel.
- Spatial mask: `s_t = sigmoid( Conv1x1( GELU( Conv3x3( H_{t-1} ) ) ) )` with
  intermediate width `spatial_gate_hidden`. The output has shape `(B, 8, 8)`
  and is applied per square.
- Selected evidence: `E_t = c_t[:, None, None] * s_t[None, :, :] * H_{t-1}`,
  the elementwise product of the two masks with the trunk feature map.
- Residual update: `H_t = GroupNorm( H_{t-1} + residual_scale * Conv3x3(E_t) )`.
  The residual conv operates on `E_t`, not `H_{t-1}`, so only the *selected*
  evidence is propagated to the next stage.

The sieve cascade is the central structural commitment of the architecture:
each stage refines the feature map, leaves a diagnostic trail of its mask,
and feeds the next stage with the sieved residue.

## Head

The classifier head consumes the across-stage mean of selected evidence
`\bar{E} = (1 / T) \sum_t E_t` *and* the final propagated trunk `H_T`:

```text
z = concat( pool(\bar{E}), pool(H_T) )
\hat{y} = Linear( GELU( Linear( LayerNorm(z) ) ) )
```

with global average pooling over the 64 squares and `hidden_dim` as the MLP
width. The head therefore depends on both the union of stage-wise selections
and on the information that survived the entire sieve cascade.

## Diagnostics

The forward pass returns a dict with the following keys (`B = batch`,
`T = num_sieves`, `C = channels`):

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` for `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `trunk_features`: `(B, C, 8, 8)` final propagated trunk `H_T`.
- `stage_selected_evidence`: `(B, T, C, 8, 8)` per-stage selected evidence
  `E_t`.
- `stage_channel_masks`: `(B, T, C)` per-stage channel masks `c_t`.
- `stage_spatial_masks`: `(B, T, 8, 8)` per-stage spatial masks `s_t`.
- `stage_selection_ratio`: `(B, T)` per-stage `mean(c_t) * mean(s_t)`.
- `stage_selected_energy`: `(B, T)` per-stage mean of `E_t^2`.
- `stage_channel_mask_entropy`: `(B, T)` Bernoulli entropy of `c_t`,
  averaged over channels.
- `stage_spatial_mask_entropy`: `(B, T)` Bernoulli entropy of `s_t`,
  averaged over the 64 squares.
- `selected_evidence_mean`: `(B, C, 8, 8)` across-stage mean `\bar{E}`.
- `selected_pool`: `(B, C)` GAP of `\bar{E}`.
- `trunk_pool`: `(B, C)` GAP of `H_T`.
- `mean_selection_ratio`: `(B,)` mean of `stage_selection_ratio`.
- `mean_selected_energy`: `(B,)` mean of `stage_selected_energy`.
- `mean_channel_mask_entropy`: `(B,)` mean of `stage_channel_mask_entropy`.
- `mean_spatial_mask_entropy`: `(B,)` mean of `stage_spatial_mask_entropy`.
- `sieve_carryover_energy`: `(B,)` mean square of `H_T - E_1`, a measure of
  how much the trunk representation actually changed across the cascade.
- `depth_levels`: `(B,)` scalar tag of the configured trunk depth.
- `sieve_levels`: `(B,)` scalar tag of the configured number of sieves.

## Implementation Binding

- Registered model name: `evidence_sieve_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/evidence_sieve_network.py`
- Idea-local wrapper: `ideas/registry/i167_evidence_sieve_network/model.py`
