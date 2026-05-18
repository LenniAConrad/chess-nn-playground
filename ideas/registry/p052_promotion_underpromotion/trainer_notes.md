# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = promotion_underpromotion`
- `model.head_hidden_dim`, `model.head_dropout`, `model.gate_init`,
  `model.ablation` for the PUGP head.
- Trunk hyperparameters retain their i193 names with a `trunk_` prefix.
- `training.batch_size` stays at the i193 default 256; PUGP does **not**
  re-run the trunk per promotion-piece substitution (unlike i246), so
  it does not need the batch-size halving that i246 documents.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults the PUGP head adds: one canonicalise + 18-channel
`index_select`, one (B, 8, 8, 7) ray gather, one cumulative blocker
scan, three per-candidate-kind ``(B, 8, 16)`` token assemblies, one
LayerNorm, and two small MLPs (delta + gate). Total head parameters
~30k. Expected wall-clock overhead at B=256 is small (low single-digit
percent over i193).

## Ablation runs

Primary geometry falsifier:

```yaml
model:
  ablation: pseudo_only
```

Additional ablations:

- `model.ablation: no_capture`         -- capture-promotion falsifier
- `model.ablation: queen_only`         -- underpromotion-hint falsifier
- `model.ablation: no_attack_defense`  -- arrival-safety falsifier
- `model.ablation: zero_delta`         -- i193 baseline
- `model.ablation: trunk_only`         -- semantic alias of zero_delta
- `model.ablation: disable_gate`       -- gate load-bearing check

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns `pugp_total_count`, `pugp_queen_check_count`
and `pugp_knight_fork_max` should be non-zero on the merged
``promotion / underpromotion`` slice and near-zero elsewhere; the
gate's mean activation on the promotion slice should rise above its
mean activation on the global set after training.
