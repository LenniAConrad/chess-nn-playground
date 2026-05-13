# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). Matches the i193
baseline on split, seed, encoding, training budget, and threshold-
selection rule.

Differences vs the i193 baseline:

- `model.name = event_symmetric_interaction_accumulator`
- `model.token_dim`, `model.order`, `model.head_hidden_dim`,
  `model.head_dropout`, `model.normalize_by_active_count`,
  `model.ablation` for the symmetric-polynomial head.
- Trunk hyperparameters retain their i193 names with a `trunk_`
  prefix.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

The streaming recurrence walks 64 squares x R orders. For R=2 it is
cheaper than a single conv layer; for R=3 it is comparable. Throughput
should be within 5-10% of i193.

## Ablation runs

Promotion requires the falsifier ablation:

```yaml
model:
  ablation: first_order_only
```

with everything else matched. If the first-order-only run matches the
unablated run on the declared slice, the higher-order terms are not
load-bearing.

## Reports

Standard idea report (see `report_template.md`). Required slices:

- Aggregate validation and test PR AUC.
- Near-puzzle FP rate at matched recall.
- Slice PR AUC for triple-interaction tactics (fork, double-attack,
  discovered-attack triple, knight fork).
- `esia_order_<r>_magnitude` distribution vs label for r >= 2.
- Cost: params, FLOPs/MACs, throughput.
