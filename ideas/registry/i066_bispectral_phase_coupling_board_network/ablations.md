# Ablations

The model exposes its central falsifiers through `model.ablation`. Run the main config plus the seven ablation configs below.

- `none` (default): full bispectral phase + magnitude + power spectrum + cross-channel + coordinate planes + material summary.
- `magnitude_only`: keep `log(1 + |Bis|)` and the power spectrum, zero out `cos(angle Bis)` / `sin(angle Bis)`. Central falsifier for phase coupling.
- `power_only`: zero both the bispectrum and the cross-channel phase terms, leaving only the power spectrum + material summary. Tests whether third-order coupling matters at all.
- `phase_batch_shuffle`: roll the bispectral phase features across the batch so they no longer line up with their own labels. Tests whether phase carries per-position evidence.
- `random_frequency_pairs`: replace the structured `(k, l)` list with a deterministic random fixed set of the same count. Tests whether structured frequency-pair selection matters.
- `channel_pair_shuffle`: roll the cross-channel phase pairs across channels so the cross-power features no longer match the channel pairing. Tests whether channel semantics matter.
- `no_coordinate_planes`: drop the deterministic coordinate planes from the mixer input. Tests whether absolute-board context is necessary.

Compare against LC0 BT4, NNUE, and the strongest registered idea runs (in particular `wavelet_scattering_board_network` and the strongest CNN baselines) on the same split and seeds to isolate the bispectral phase-coupling contribution.
