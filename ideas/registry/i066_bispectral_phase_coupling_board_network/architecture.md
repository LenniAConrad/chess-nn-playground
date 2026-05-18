# Architecture

`Bispectral Phase-Coupling Board Network` realises the source packet's Fourier bispectral phase-coupling classifier as a bespoke PyTorch model for the repository's `puzzle_binary` task. The classifier never sees raw board planes; it reads the position only through a deterministic 2D FFT of learned `1x1` channel mixtures, summarised by selected bispectrum, power-spectrum, and cross-channel phase features.

## Implementation Binding

- Registered model name: `bispectral_phase_coupling_board_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/bispectral_phase_coupling.py`
- Idea-local wrapper: `ideas/registry/i066_bispectral_phase_coupling_board_network/model.py`

## Modules

`SpectralChannelMixer` is a `1x1` convolution from the 18 `simple_18` planes to `Cmix=16` mixed real board fields. When `use_coordinate_planes=True` (default) the mixer is fed four deterministic side-relative coordinate planes in addition to the board planes — rank, file, center distance, and side-to-move-relative forward — so the mixed fields can absorb absolute-board context that translation-invariant bispectral terms drop. Filter weights are the only learnable parameters in the front end.

`BoardFFTFeatureLayer` runs `torch.fft.fft2` over the last two axes of the mixed planes, producing a `(B, Cmix, 8, 8)` complex grid. The transform is deterministic and frozen.

`BispectralPhaseCoupling` reads complex coefficients for a fixed list of `T=48` frequency pairs `(k, l)` and forms the bispectrum
`Bis(k, l) = F(k) * F(l) * conj(F(k + l mod (8, 8)))`. The pair list is generated deterministically by `_structured_frequency_pairs` (eight directional pairs followed by a low x low cross-product), or by `_random_fixed_frequency_pairs` when the `random_frequency_pairs` ablation is selected. Per bispectrum value it emits three features:

- `cos(angle(Bis))` and `sin(angle(Bis))` from the unit-norm phase factor,
- `log(1 + |Bis|)` as a stable magnitude summary.

When `include_power_spectrum=True` the module appends `log(1 + |F(k)|^2)` for 16 low-frequency coordinates per mixed channel. When `include_cross_channel_phase=True` it appends cross-power phase and magnitude features `F_a(k) * conj(F_b(k))` for 8 adjacent channel pairs over 12 low-frequency indices. A 20-dimensional material summary (side-relative piece counts, count delta, total count, and material balance) is concatenated last so the classifier still sees ordinary material context. The shared 1x1 mixer / coordinate planes contract is enforced by `BoardTensorSpec(input_channels=18)`.

The full feature vector at the default config has dimension `16 * 48 * 3 + 16 * 16 + 8 * 12 * 3 + 20 = 2868`. This is the only signal that reaches the classifier head.

`BispectralPhaseHead` is a `LayerNorm + (Linear + GELU + Dropout) x 2 + Linear(num_classes)` MLP. For `num_classes=1` the head produces one BCE-compatible logit per board.

`BispectralPhaseCouplingBoardNetwork` glues the trunk together: `SpectralChannelMixer -> BoardFFTFeatureLayer -> BispectralPhaseCoupling -> BispectralPhaseHead`. The `ablation` argument selects the active variant:

- `none` (default): full bispectral phase + magnitude + power spectrum + cross-channel features.
- `magnitude_only`: zeros the bispectral phase features but keeps `log(1 + |Bis|)` and the power spectrum; central falsifier for phase coupling.
- `power_only`: zeros both the bispectrum and the cross-channel phase terms, leaving only the power spectrum + material summary; tests whether third-order coupling matters at all.
- `phase_batch_shuffle`: rolls the bispectral phase features across the batch so they no longer line up with their own labels.
- `random_frequency_pairs`: replaces the structured `(k, l)` list with a deterministic random fixed set with the same count.
- `channel_pair_shuffle`: rolls the cross-channel phase pairs across channels so the cross-power features no longer match the channel pairing.
- `no_coordinate_planes`: drops the deterministic coordinate planes from the mixer input.

## Diagnostics

`forward(x)` returns a dict containing:

- `logits`: shape `(B,)` (or `(B, num_classes)` for non-binary configs), BCE-compatible for the one-logit `puzzle_binary` head.
- `prob`: sigmoid of the puzzle logit (softmax when `num_classes > 1`).
- `bispectral_phase_norm`: root-mean-square of the unit-norm bispectrum phase features.
- `bispectral_magnitude_mean`: mean of `log(1 + |Bis|)`.
- `power_spectrum_energy`: mean of `log(1 + |F(k)|^2)` across mixed channels and the selected power frequencies (zero when the power spectrum is disabled).
- `cross_phase_norm`: root-mean-square of the cross-channel phase features (zero when disabled).
- `spectral_feature_norm`: per-sample norm of the full pooled feature vector.
- `mixed_field_energy`: mean square of the post-mixer real board fields.
- `material_balance`: side-relative scaled material balance scalar from the deterministic material summary.
- `mechanism_energy`: `bispectral_phase_norm^2 + bispectral_magnitude_mean^2`, the bispectrum energy that operationalises the packet's `linear_algebra` mechanism family.
- `proposal_profile_strength`: per-sample max of phase-norm vs magnitude-mean.
- `proposal_keyword_count`: scalar count of active feature groups (phase, magnitude, power, cross-channel, material, coordinate) preserved for compatibility with the project's research-packet diagnostic schema.
- `bispectral_ablation`: integer code identifying the active ablation mode.
- `bispectral_term_count`: scalar reporting the bispectrum term count `T`.

## Contract

- Input: `(B, 18, 8, 8)` board tensor only. CRTK / verification / source / engine metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit `puzzle_binary` BCE-with-logits trainer, plus the diagnostics listed above.
- Target mapping: fine labels `0` and `1` map to binary target `0`; fine label `2` maps to binary target `1`.
- Model shapes: mixed planes `[B, Cmix, 8, 8]`, FFT coefficients `[B, Cmix, 8, 8]` complex, pooled features `[B, 2868]` at the default config.
- The puzzle decision flows only through the pooled spectral feature vector — the head never sees raw board planes, so the bispectral bottleneck is enforced architecturally.
- The FFT and frequency-pair tables are stored in non-persistent buffers and never optimised; only the `1x1` mixer, `LayerNorm`, and head linears are trainable.
