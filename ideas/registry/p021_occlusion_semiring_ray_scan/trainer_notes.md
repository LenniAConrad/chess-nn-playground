# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). Matches the i193
baseline on split, seed, encoding, training budget, and threshold-
selection rule.

Differences vs the i193 baseline:

- `model.name = occlusion_semiring_ray_scan`
- `model.token_dim`, `model.hidden_dim`, `model.head_hidden_dim`,
  `model.head_dropout`, `model.ablation` for the ray-scan head.
- Trunk hyperparameters retain their i193 names with a `trunk_`
  prefix.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

The exclusive prefix transmittance is `O(8 * 64 * 7)` per sample
(small compared to the conv trunk). The dominant head cost is the
per-direction projection (`O(8 * 64 * 7 * token_dim * hidden_dim)`)
which is comparable to one additional conv mixing layer. Throughput
should be within ~15% of i193.

## Ablation runs

Promotion requires the falsifier ablation:

```yaml
model:
  ablation: zero_occupancy
```

with everything else matched. If the no-blocker run matches the
unablated run on the declared slice, the transmittance is not
load-bearing and the primitive should be dropped.

## Reports

Standard idea report (see `report_template.md`). Required slices:

- Aggregate validation and test PR AUC.
- Near-puzzle FP rate at matched recall.
- Slice PR AUC for x-ray / pin / skewer / discovered-attack /
  blocker-dependent tactics.
- `osrs_mean_transmittance` and `osrs_open_ray_fraction` distribution
  vs label.
- Cost: params, FLOPs/MACs, throughput.
