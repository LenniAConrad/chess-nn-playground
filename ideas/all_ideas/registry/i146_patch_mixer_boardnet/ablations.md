# Ablations

- `model.ablation: patch1_square_mixer` uses `1 x 1` square tokens instead of `2 x 2` patches.
- `model.ablation: patch4_coarse_mixer` uses `4 x 4` coarse board patches.
- `model.ablation: no_token_mixing` removes cross-patch token mixing.
- `model.ablation: no_channel_mixing` removes per-patch channel mixing.
- `model.ablation: cnn_matched_params` swaps in a plain CNN control with similar width and depth.
- Reduce `model.depth` to 1 to test whether the Mixer survives a smaller stack.
- Compare against LC0 BT4, NNUE, and the strongest registered idea runs on the same split and seeds.
