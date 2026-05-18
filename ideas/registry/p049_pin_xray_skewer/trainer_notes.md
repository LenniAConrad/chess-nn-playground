# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold rule).

Differences vs i193:

- `model.name = pin_xray_skewer`
- `model.head_hidden_dim`, `model.head_dropout`, `model.gate_init`,
  `model.ablation` for the PXS head
- trunk hyperparameters retain their i193 names with a `trunk_` prefix
- ``training.batch_size`` stays at the i193 default 256 (the head adds
  trivial memory; the event builder does not allocate any pairwise
  `(64, 64)` tensors, only per-source-square `(B, 6, 64)`)

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (head_hidden_dim=64, B=256) the per-step overhead of the
PXS head is small compared to the trunk: the event builder touches a
`(B, 8, 64, 7)` tensor 8 times for the scalar gathers and runs one
`cumsum` along the 7-step axis. There are no Python loops in the
forward path. Throughput should be within +10% of the i193 baseline.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: no_xray1
```

Additional ablations:

- `model.ablation: uniform_values`  -- value-context falsifier
- `model.ablation: no_pin_def`      -- defender-load load-bearing
- `model.ablation: shuffle_rays`    -- geometry decoupling
- `model.ablation: zero_delta`      -- i193 baseline
- `model.ablation: trunk_only`      -- strongest control
- `model.ablation: disable_gate`    -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns ``primitive_gate``, ``primitive_delta``,
``pxs_abs_pin_mean``, ``pxs_skewer_mean``, ``pxs_discovered_mean`` and
``pxs_pinned_defender_mean`` should be inspected to confirm the gate
fires preferentially on positions with high ordered-blocker structure
(absolute pins, skewers, discovered attacks). The unablated run
should show non-trivial mass on at least the `abs_pin` and `skewer`
channels for the `pin` and `skewer` motif slices.
