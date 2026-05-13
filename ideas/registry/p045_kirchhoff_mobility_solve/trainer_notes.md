# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = kirchhoff_mobility_solve`
- `model.source_channels`, `model.output_channels`,
  `model.head_hidden_dim`, `model.head_dropout`, `model.gate_init`,
  `model.shift`, `model.ablation` for the KMS head.
- trunk hyperparameters retain their i193 names with a `trunk_` prefix.
- ``training.batch_size`` remains at the i193 default 256; the SPD
  solve is small enough not to dominate memory.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (``source_channels=6``, ``output_channels=8``, B=256),
throughput should be within +20% of the i193 baseline. The dominant
cost is the single batched SPD solve of size 64x64.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: uniform_conductance
```

Additional ablations:

- `model.ablation: diagonal_only`       -- drop the Laplacian entirely
- `model.ablation: shuffle_conductance` -- rule-feature falsifier
- `model.ablation: zero_source`         -- sanity check
- `model.ablation: zero_delta`          -- i193 baseline
- `model.ablation: trunk_only`          -- strongest control
- `model.ablation: disable_gate`        -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns ``kms_potential_norm`` and
``kms_conductance_mean`` should be inspected to confirm the
conductance does not collapse to a constant (which would render the
``uniform_conductance`` ablation indistinguishable from the
unablated model).
