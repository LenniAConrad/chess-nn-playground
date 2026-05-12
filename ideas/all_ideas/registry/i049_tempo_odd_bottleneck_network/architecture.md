# Architecture

`Tempo-Odd Bottleneck Network` is a board-only `puzzle_binary`
classifier whose central operator is a deterministic side-to-move
involution `tau` paired with a two-point Walsh odd/even projection of a
shared convolutional encoder. The implementation replaces the shared
research-packet probe with a materially distinct bespoke model so the
markdown thesis is exercised by trainable code rather than a generic
mechanism profile.

## Forward Pipeline

1. **Adapter / counterfactual builder.** The `simple_18` board tensor
   `(B, 18, 8, 8)` is validated and conservatively sanitized: the
   en-passant plane (channel 17) is zeroed in both the real view and
   the counterfactual to keep `tau` rule-clean. The deterministic
   intervention `tau` flips only the side-to-move plane (channel 12),
   producing the side-to-move twin. The two views are concatenated
   along the batch dimension into `(2B, 18, 8, 8)`. No move
   generation, mate flag, engine input, CRTK source label, or
   verification metadata is consulted. Unsupported encodings fail
   closed.
2. **Shared board encoder.** A compact convolutional tower of
   `Conv(18 -> width)` stem plus `encoder_blocks` residual blocks at
   `width` channels with GELU activations, followed by global average
   pooling and a `LayerNorm/Linear/GELU` projection to `latent_dim`,
   is applied to the concatenated `(2B, 18, 8, 8)` batch and split back
   into `h0` and `h_tau` of shape `(B, latent_dim)`.
3. **Two-point Walsh odd/even split.** The architecture computes
   `z_even = 0.5 * (h0 + h_tau)` and `z_odd = 0.5 * (h0 - h_tau)`. By
   the involution identity `tau(tau(x)) = x` the even projection is
   `tau`-invariant and the odd projection is `tau`-anti-invariant; if
   the encoder admits the decomposition `h(B, s, R) = u(B, R) + s v(B, R)`
   with `s in {-1, +1}`, the odd projection exactly cancels any
   side-blind term and exactly recovers the first-order side-to-move
   interaction.
4. **Bottleneck projections.** The high-capacity predictive path is the
   odd branch: `LayerNorm(z_odd)` is mapped through a no-bias linear
   projection to `odd_dim`, producing `odd_signed`, and its element-wise
   absolute value `odd_magnitude`. The low-capacity context path is the
   even branch: `LayerNorm(z_even)` is routed through a small linear
   projection to `even_dim`, with stop-gradient applied by default so
   classifier gradients cannot drive the shared encoder through the
   side-blind context.
5. **Classifier head.** The head receives `cat([odd_signed,
   odd_magnitude, even_context])` and applies a `Linear -> GELU ->
   Dropout -> Linear` MLP returning two-class logits. For the
   `puzzle_binary` BCE-with-logits trainer the head emits the scalar
   `logit_1 - logit_0` so the output `logits` has shape `(B,)`; the
   raw two-class logits are also returned for diagnostic use.

## Output Contract

Forward returns a `dict` whose `"logits"` entry has shape `(B,)` for the
puzzle_binary BCE-with-logits trainer when `num_classes=1` (or `(B, 2)`
for cross-entropy when `num_classes=2`). Diagnostic tensors include
`two_class_logits`, `tempo_odd_norm`, `tempo_even_norm`, `odd_energy`,
`even_energy`, `odd_to_even_energy_ratio`, `side_intervention_gap`,
`odd_variance_loss`, and `en_passant_removed`. All diagnostic tensors
are finite by construction.

## Implementation Binding

- Registered model name: `tempo_odd_bottleneck_network`
- Source implementation file: `src/chess_nn_playground/models/tempo_odd_bottleneck.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i049_tempo_odd_bottleneck_network/model.py`
