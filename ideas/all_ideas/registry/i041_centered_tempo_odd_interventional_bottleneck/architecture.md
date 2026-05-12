# Architecture

`Centered Tempo-Odd Interventional Bottleneck` is a board-only
`puzzle_binary` classifier whose central operator is a deterministic
C2 side-to-move involution combined with a null-board centering of the
encoder's pure-turn response. The implementation replaces the shared
research-packet probe with a materially distinct bespoke model so the
markdown thesis is exercised by trainable code rather than a generic
mechanism profile.

## Forward Pipeline

1. **Adapter / counterfactual builder.** The simple_18 board tensor
   `(B, 18, 8, 8)` is validated and three deterministic counterfactual
   tensors are constructed: the side-to-move twin `tau(x)` (only the
   side-to-move plane at channel 12 is flipped), the null board `nu(x)`
   (every plane except the side-to-move plane is zeroed) and its
   toggled twin `tau(nu(x))`. The four views are concatenated along the
   batch dimension into `x4` of shape `(4B, 18, 8, 8)`. No move
   generation, mate flag, engine input, CRTK source label or
   verification metadata is consulted. Unsupported encodings fail
   closed.
2. **Shared board encoder.** A compact convolutional tower
   `Conv(18 -> 64) -> norm/GELU -> Conv(64 -> 96) -> norm/GELU` followed
   by `encoder_blocks` residual blocks at width 96 is applied to the
   concatenated batch. Output shape: `(4B, 96, 8, 8)` with no spatial
   down-sampling so the centered odd map preserves the 8x8 board grid.
3. **C2 odd/even split.** The encoded batch is split back into
   `h, h_tau, h_null, h_null_tau`. The model computes the side-to-move
   anti-invariant component `odd = 0.5 * (h - h_tau)`, the symmetric
   component `even = 0.5 * (h + h_tau)`, and the null-board odd
   component `null_odd = 0.5 * (h_null - h_null_tau)`. The classifier
   feature is `centered_odd = odd - null_odd`, which by construction is
   anti-invariant under turn toggling and removes the additive
   board-only term, the constant term, and the pure-turn offset. The
   `even` map is computed for diagnostics but is not consumed by the
   main head.
4. **Pooling head.** Spatial mean, max and RMS of `centered_odd` are
   stacked into a `(B, 3 * 96)` feature vector. A two-layer MLP
   (`3 * 96 -> hidden -> num_classes`) with GELU activation, dropout,
   and head hidden width 192 returns the puzzle logit(s).

## Output Contract

Forward returns a `dict` whose `"logits"` entry has shape `(B,)` for the
puzzle_binary BCE-with-logits trainer when `num_classes=1` (or `(B, 2)`
for cross-entropy when `num_classes=2`). Diagnostic tensors include
`tempo_odd_norm`, `tempo_even_norm`, `null_odd_norm`,
`centered_odd_norm`, `side_intervention_gap`, `centered_odd_energy`,
and the `centered_odd` field itself. All diagnostic tensors are finite
by construction.

## Implementation Binding

- Registered model name: `centered_tempo_odd_interventional_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/centered_tempo_odd_interventional_bottleneck.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i041_centered_tempo_odd_interventional_bottleneck/model.py`
