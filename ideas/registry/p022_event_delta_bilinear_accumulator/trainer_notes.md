# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). Matches the i193
baseline on split, seed, encoding, training budget, and threshold-
selection rule.

Differences vs the i193 baseline:

- `model.name = event_delta_bilinear_accumulator`
- `model.bilinear_dim`, `model.head_hidden_dim`,
  `model.head_dropout`, `model.normalize_by_active_count`,
  `model.ablation` for the bilinear head.
- Trunk hyperparameters retain their i193 names with a `trunk_`
  prefix.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

Two linear projections on `(B, 64, 13)` tokens plus a constant number
of Hadamard sums. Throughput should be within 5% of i193.

## Ablation runs

Promotion requires the falsifier ablation:

```yaml
model:
  ablation: first_order_only
```

with everything else matched. If the no-pair-term run matches the
unablated run on the declared slice, the bilinear term is not load-
bearing.

## Reports

Standard idea report (see `report_template.md`). Required slices:

- Aggregate validation and test PR AUC.
- Near-puzzle FP rate at matched recall.
- Slice PR AUC for second-order interaction tactics (king-piece,
  bishop pair, mutual defense, overloaded defender).
- `edba_pair_term_magnitude` distribution vs label.
- Cost: params, FLOPs/MACs, throughput.
