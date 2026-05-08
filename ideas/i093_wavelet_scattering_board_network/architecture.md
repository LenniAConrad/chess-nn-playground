# Architecture

`Wavelet Scattering Board Network` realises the source packet's fixed multiscale wavelet scattering front end as a bespoke PyTorch model for the repository's `puzzle_binary` task. The classifier never sees raw board planes; it reads the position only through fixed Haar wavelet modulus statistics across scale, orientation, and channel.

## Implementation Binding

- Registered model name: `wavelet_scattering_board_network`
- Source implementation file: `src/chess_nn_playground/models/wavelet_scattering_board_network.py`
- Idea-local wrapper: `ideas/i093_wavelet_scattering_board_network/model.py`

## Modules

`FixedWaveletBank` is a depthwise 2x2 wavelet operator at a fixed dilation `s in {1, 2, 4}`. Its weight buffer is a non-learnable replicated copy of the four 2x2 Haar filters `(LL, H, V, D)` (or four random orthogonal filters, when ablating). It applies `F.conv2d` with `groups=input_channels` after circular padding, so a board tensor of shape `(B, 18, 8, 8)` becomes `(B, 18, 4, 8, 8)` with the fast 4 axis carrying `(LL, H, V, D)`.

`WaveletScatteringFeatures` stacks one `FixedWaveletBank` per scale into a first scattering layer. The forward pass produces

- `LL_{c, s}`: the scale-`s` lowpass response per channel `c`,
- `U^{(1)}_{c, s, o} = | psi_{s, o} * x_c |`: the first-order modulus fields for the three high-pass orientations `o in {H, V, D}`.

When `second_order=True` (default) a second `FixedWaveletBank` per scale is applied to the first-order modulus fields. Following the standard scattering tree, only second-scale indices `s2 > s1` are kept and the modulus is taken on the high-pass output, giving `U^{(2)}_{c, s1, o1, s2, o2}`.

`WaveletScatteringFeatures` then pools the fields into a fixed feature vector:

- First-order stats per `(c, s, o)`: `mean`, `std`, and `max` of `U^{(1)}`.
- Lowpass signed energy per `(c, s)`: `mean(LL_{c, s})`.
- Second-order means per `(c, s1, o1, s2, o2)` with `s2 > s1`: `mean(U^{(2)})`.

For `C = 18`, three scales, and three orientations the feature count is `18 * (3*9 + 3 + 3*9) = 1026`. This is the only signal that reaches the classifier head.

`WaveletScatteringBoardNetwork` glues the trunk together: optional fixed channel permutation -> `WaveletScatteringFeatures` -> `LayerNorm + (Linear + GELU + Dropout) x depth + Linear(1)` head. The head returns one logit per board; the rest of the diagnostics surface as named tensors on the output dict.

The `mode` argument selects the active variant:

- `haar` (default): fixed 2x2 Haar filters at dilations 1, 2, 4 with two-layer scattering. The reference implementation called for in the source packet.
- `random_fixed_filters`: replaces the Haar bank at every scale with a fixed random orthogonal 2x2 bank. Tests whether wavelet structure is what is helping.
- `lowpass_only`: zeroes out every first-order modulus field (and so every second-order field), leaving only the per-scale lowpass signed energies. Tests whether multiscale edges add anything beyond running averages.
- `channel_shuffle`: applies a fixed permutation to the input channels before scattering. Tests whether semantic piece planes matter at all.

## Diagnostics

`forward(x, *, return_diag=False)` returns a dict containing:

- `logits`: shape `(B,)`, BCE-compatible for the one-logit `puzzle_binary` head.
- `prob`: sigmoid of the puzzle logit.
- `scattering_features`: the full pooled feature vector phi(x) used by the head, shape `(B, 1026)` at the default config.
- `first_order_mean_field`, `first_order_std_field`, `first_order_max_field`: shape `(B, C, S, O)` per-channel/scale/orientation pool stats.
- `lowpass_energy`: shape `(B, C, S)` signed lowpass energies.
- `second_order_mean_field` (when `second_order=True`): flat `(B, C * pairs * O * O)` second-order modulus means.
- `scattering_mode`: integer code identifying the active mode (`haar`/`random_fixed_filters`/`lowpass_only`/`channel_shuffle`).
- `mechanism_energy`: `mean( first_order_mean_field^2 )` — the multiscale modulus energy that operationalises the packet's `linear_algebra` mechanism family.
- `proposal_profile_strength`: the largest first-order maximum across `(c, s, o)`.
- `proposal_keyword_count`: integer scalar preserved for compatibility with the project's research-packet diagnostic schema.
- `scale_count`: integer scalar reporting the number of scales used.

When `return_diag=True` the dict additionally contains `first_order_modulus` of shape `(B, C, S, O, 8, 8)` and `lowpass_field` of shape `(B, C, S, 8, 8)` for ablation harnesses.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Model shapes: lowpass `[B, C, S, 8, 8]`, first-order modulus `[B, C, S, O, 8, 8]`, scattering features `[B, 1026]` at the default config.
- The puzzle decision flows only through phi(x) — the head never sees raw board planes, so the multiscale-modulus bottleneck is enforced architecturally.
- The wavelet filter banks are stored in non-persistent buffers and never optimised; only the `LayerNorm` and `Linear` head parameters are trainable.
