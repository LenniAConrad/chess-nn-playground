# Architecture

`Rule-Exact Orbit Bottleneck Network` (idea i046) is a board-only
`puzzle_binary` classifier whose central operator is a deterministic
Reynolds projection over the rule-exact color-flip orbit
`G = {e, kappa}` of the chess automorphism group, exactly as specified
by `math_thesis.md` and the source research packet. The shared
`ResearchPacketProbe` mechanism profile is no longer used here; the
implementation is materially distinct.

## Forward Pipeline

1. **`Simple18ColorFlipAdapter`.** A deterministic, parameter-free
   adapter constructs the orbit `{x, kappa(x)}` from the
   `simple_18` board tensor. The exact color-flip transform `kappa`
   performs (a) a vertical rank mirror, (b) a swap of the white and
   black piece occupancy planes (channels 0..5 and 6..11), (c) a
   complement of the side-to-move plane (channel 12), (d) the
   `KQkq -> kqKQ` swap of the four castling-rights planes
   (channels 13..16), and (e) the rank mirror of the en-passant plane
   (channel 17). With `orbit_group="rank_flip_no_color"` the adapter
   instead applies a vertical rank flip only, which is the central
   semantics-destroying falsifier specified by the research packet.
   The adapter fails closed if the channel schema is anything other
   than `simple_18`/18 channels unless explicitly opted out.
2. **Shared `TinyBoardStem`.** A compact convolutional encoder shared
   across orbit views: a `3x3` Conv stem followed by `num_blocks`
   residual blocks (`ResidualMicroBlock`, two `3x3` convolutions with
   GroupNorm + GELU each), then a `1x1` projection to width
   `2 * stem_width` and global average pooling to `(B*|G|, 2*stem_width)`.
3. **Shared latent projection.** `Linear(2*stem_width, latent_dim)`
   with LayerNorm + GELU, applied to every orbit view.
4. **Per-view classifier.** A single shared `Linear(latent_dim,
   num_classes)` produces per-view logits of shape `(B, |G|,
   num_classes)`.
5. **Reynolds pooling.** The final classifier output is the Reynolds
   projection `R_G[f](x)` over the orbit, controlled by `pool_mode`:
   - `probability_mean` (default, matching the math thesis): average
     per-view sigmoid/softmax probabilities and emit
     `logit(p_bar)` as the model's binary logit.
   - `logit_mean`: average per-view logits.
   - `latent_mean`: average per-view latents and apply the classifier
     once.
   For `num_classes=1` the model emits a single binary puzzle logit;
   for `num_classes=2` it emits a two-class log-probability vector.

## Output Contract

`forward(x)` returns a `dict` keyed by:

- `logits`: shape `(B,)` for `num_classes=1` (single-logit BCE used
  by the puzzle-binary contract) or `(B, 2)` for `num_classes=2`.
- `identity_view_logit`, `transformed_view_logit`: per-view puzzle
  logits at the identity and transformed orbit views.
- `view_logit_gap`: `|identity_view_logit - transformed_view_logit|`,
  the symmetry-residual logit defect.
- `identity_probability`, `transformed_probability`: per-view
  puzzle-likeness probabilities.
- `mean_view_probability`: the orbit-averaged puzzle probability
  (Reynolds-pooled mean, before logit).
- `orbit_probability_gap`, `symmetry_residual`: absolute orbit
  probability defect, the falsification observable from the math
  thesis.
- `latent_orbit_variance`: per-sample variance of the orbit latents,
  a diagnostic of how strongly the shared encoder breaks invariance.
- `mechanism_energy`: mean squared latent magnitude.
- `orbit_size`: scalar `|G|` per sample, exposed for diagnostics and
  the falsifier comparison (`identity` vs `color_flip` vs
  `rank_flip_no_color`).

These auxiliary tensors are exactly the falsification observables
listed in `math_thesis.md` and `architecture.md`'s research packet:
the orbit probability gap is the symmetry residual whose sign and
magnitude separate the rule-exact color flip from the
`rank_flip_no_color` ablation.

## Falsifier Wiring

The packet's central falsifier is `kappa -> rank_flip_no_color`.
The same model class supports it directly through
`orbit_group="rank_flip_no_color"`, which leaves the stem, head, and
parameter count unchanged. `orbit_group="identity"` recovers the
single-view ablation (no orbit pooling, same compute as the
non-orbit baseline). All three branches share the `TinyBoardStem`
and classifier so that any difference in puzzle-binary metrics
isolates the orbit operator rather than capacity or augmentation.

## Implementation Binding

- Registered model name: `rule_exact_orbit_bottleneck_network`
- Source implementation file: `src/chess_nn_playground/models/rule_exact_orbit_bottleneck.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i046_rule_exact_orbit_bottleneck_network/model.py`
