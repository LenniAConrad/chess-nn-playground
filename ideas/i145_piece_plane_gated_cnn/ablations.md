# Ablations

- `model.ablation: ungrouped_stem_matched` replaces the semantic group stems with one matched stem over all planes.
- `model.ablation: no_gates` keeps the group stems but uses unit gates.
- `model.ablation: random_channel_groups` preserves group sizes but assigns planes by a fixed random permutation.
- Reduce `model.trunk_depth` or `model.stem_depth` to test whether the mechanism survives a smaller trunk.
- Compare against LC0 BT4, NNUE, and the strongest registered idea runs on the same split and seeds.
