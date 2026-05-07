# Ablations

- `single_global_head`: use only the global pooling head.
- `no_king_head`: remove the king-zone specialist.
- `no_material_head`: remove the material/count specialist.
- `uniform_logit_average`: average active specialist logits instead of using
  the learned fusion MLP.
- `same_region_random_masks`: replace center and edge masks with deterministic
  random masks of the same sizes.

Compare each ablation against the default `specialist_head_cnn` config on the
same puzzle_binary split and seeds.
