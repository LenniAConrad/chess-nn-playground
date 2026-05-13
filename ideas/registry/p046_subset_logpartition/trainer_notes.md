# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = subset_logpartition`
- `model.log_weight_dim`, `model.degree`, `model.head_hidden_dim`,
  `model.head_dropout`, `model.gate_init`, `model.ablation` for the
  SLPT head.
- trunk hyperparameters retain their i193 names with a `trunk_` prefix.
- ``training.batch_size`` stays at the i193 default 256; the SLPT
  scan does not allocate pair tensors.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (``log_weight_dim=32``, ``degree=3``, B=256) the per-step
overhead of the SLPT head is small compared to the trunk: a Python
loop over 64 squares with K=3 ``logaddexp`` ops per token, over
(B, log_weight_dim) tensors. Throughput should be within +10% of the
i193 baseline.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: k1_only
```

Additional ablations:

- `model.ablation: uniform_mask`     -- rule-mask-removal control
- `model.ablation: shuffle_mask`     -- rule-feature falsifier
- `model.ablation: shuffle_tokens`   -- order-invariance check
- `model.ablation: zero_delta`       -- i193 baseline
- `model.ablation: trunk_only`       -- strongest control
- `model.ablation: disable_gate`     -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns ``slpt_y2`` and ``slpt_y3`` (per-degree log-
partition means) should differ from ``slpt_y1`` on tactically active
positions; otherwise the K>=2 information is not informative.
