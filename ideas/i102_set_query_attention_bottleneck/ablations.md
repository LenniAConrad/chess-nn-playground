# Ablations

- Set `model.ablation: uniform_attention` to replace learned attention with uniform token averaging.
- Set `model.ablation: random_frozen_queries` to test whether learned tactical queries matter.
- Set `model.ablation: value_only_no_diagnostics` to remove attention-shape diagnostics from the head.
- Set `model.ablation: diagnostics_only` to classify from entropy, margin, mass, and coordinate statistics only.
- Set `model.ablation: mean_pool_matched_params` to compare against ordinary set pooling in the same head surface.
- Compare against LC0 BT4, NNUE, and the strongest registered idea runs on the same split and seeds.
