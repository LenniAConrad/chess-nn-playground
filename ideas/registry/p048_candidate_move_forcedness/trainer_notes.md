# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config
mirrors the i193 baseline (same split, encoding, seed, budget,
threshold-rule).

Differences vs i193:

- `model.name = candidate_move_forcedness`
- `model.token_dim`, `model.score_hidden_dim`,
  `model.head_hidden_dim`, `model.head_dropout`, `model.topk`,
  `model.softmax_temperature`, `model.gate_init`, `model.ablation`
  for the CMF head.
- Trunk hyperparameters retain their i193 names with a `trunk_`
  prefix.
- `training.batch_size` stays at the i193 default 256; the dense
  `(B, 64, 64, 14)` edge feature tensor fits at this batch size.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (`token_dim=24, score_hidden_dim=32, topk=4, B=256`)
the per-step overhead is dominated by the score MLP forward over
the dense `(B * 64 * 64, ...)` flattened input and the
deterministic edge-feature builder. Throughput should be within
+15% of the i193 baseline. The legal-move graph helper is
analytic / vectorised (no Python loop) and is the smaller cost.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: deterministic_score
```

Additional ablations:

- `model.ablation: mean_pool`            -- anti-top-k
- `model.ablation: flags_only`           -- anti-deep-features
- `model.ablation: dense_edges`          -- anti-legal-mask
- `model.ablation: no_consequence`       -- anti-check/capture/promotion
- `model.ablation: zero_delta`           -- i193 baseline
- `model.ablation: trunk_only`           -- strongest control
- `model.ablation: disable_gate`         -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns `cmf_top1_score`, `cmf_gap12`,
`cmf_check_peak`, and `cmf_capture_peak` should vary across
positions; if they collapse to constant per-board values on the
test split, the head has not learned candidate-level structure and
the `deterministic_score` ablation should match the unablated run.
