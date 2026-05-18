# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config
mirrors the i193 baseline (same split, encoding, seed, budget,
threshold rule).

Differences vs i193:

- `model.name = defender_overload_triad`
- `model.head_hidden_dim`, `model.head_dropout`, `model.gate_init`,
  `model.ablation` for the DOT head
- trunk hyperparameters retain their i193 names with a `trunk_` prefix
- ``training.batch_size`` stays at the i193 default 256 (the head
  adds modest memory; the overload core never materialises a
  `(B, N, N, N)` triple tensor, only `(B, N, N)` BMMs)

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (head_hidden_dim=64, B=256) the per-step overhead of the
DOT head is small compared to the trunk. The attack builder does a
batched `(B, 64, 64)` `einsum` per side; the overload core does two
`(B, 64, 64) x (B, 64, 1)` BMMs per side plus a small MLP over a
`(B, 64, 8)` feature tensor. Throughput should be within +15% of the
i193 baseline -- a hair slower than p049 because of the dense
attack-mask construction.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: no_cross_target_load
```

Additional ablations:

- `model.ablation: no_pins`           -- pin load-bearing test
- `model.ablation: no_target_value`   -- value-weighting falsifier
- `model.ablation: counts_only`       -- SEE-light feature falsifier
- `model.ablation: zero_delta`        -- i193 baseline (numeric recovery)
- `model.ablation: trunk_only`        -- strongest control
- `model.ablation: disable_gate`      -- gate load-bearing

## Reports

Standard idea report; required slices listed in
`report_template.md`. The diagnostic columns ``primitive_gate``,
``primitive_delta``, ``overload_us_peak``, ``overload_them_peak``,
``overload_defender_burden_us``, ``overload_defender_burden_them``,
and ``overload_pinned_share_*`` should be inspected to confirm the
gate fires preferentially on positions with high overload signal
(multiple critical targets sharing a defender). The unablated run
should show non-trivial defender_burden on the ``overload``,
``deflection``, and ``pin`` motif slices.
