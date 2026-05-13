# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = octilinear_selective_scan`
- `model.feature_dim`, `model.head_hidden_dim`, `model.head_dropout`,
  `model.gate_init`, `model.ablation` for the OSS head
- trunk hyperparameters retain their i193 names with a `trunk_` prefix

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

Eight per-direction selective scans implemented as Python loops over 8
sequential steps. The wall-clock overhead is materially higher than
the i193 baseline -- watch ``train_samples_per_second`` and prefer
``feature_dim=16`` (the default) for the first scout. If throughput
drops by more than 50%, halve ``feature_dim`` before further scouts
or pursue the Triton-kernel upgrade documented in
``implementation_notes.md``.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: single_direction
```

Additional ablations:

- `model.ablation: fixed_transition` -- data-dependent selectivity test
- `model.ablation: shuffle_features`  -- rule-feature falsifier
- `model.ablation: zero_delta`        -- i193 baseline
- `model.ablation: trunk_only`        -- strongest control
- `model.ablation: disable_gate`      -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns ``primitive_gate``, ``primitive_delta``, and
``oss_energy_<direction>`` should be inspected to confirm which scan
direction dominates the per-position activation.
