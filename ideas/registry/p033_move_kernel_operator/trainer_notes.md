# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = move_kernel_operator`
- `model.feature_dim`, `model.head_hidden_dim`, `model.head_dropout`,
  `model.gate_init`, `model.ablation` for the MKO head
- trunk hyperparameters retain their i193 names with a `trunk_` prefix

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

6 per-type linear projections + one batched einsum on (T, 64, 64) x
(B, T, 64, d). With ``feature_dim=24`` the head wall-clock stays close to
the i193 baseline.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: shared_kernel
```

Additional ablations:

- `model.ablation: scalar_per_type`  -- matrix vs scalar capacity
- `model.ablation: shuffle_features` -- rule-feature falsifier
- `model.ablation: zero_delta`       -- i193 baseline
- `model.ablation: trunk_only`       -- strongest control
- `model.ablation: disable_gate`     -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns ``primitive_gate``, ``primitive_delta``, and
``mko_norm_<name>`` should be inspected to confirm which move type
dominates the per-position activation.
