# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = truncated_multiset_polynomial_pool`
- `model.latent_dim`, `model.degree`, `model.head_hidden_dim`,
  `model.head_dropout`, `model.gate_init`, `model.coeff_norm`,
  `model.ablation` for the TMPP head
- trunk hyperparameters retain their i193 names with a `trunk_` prefix
- ``training.batch_size`` stays at the i193 default 256 (the head adds
  trivial memory; the polynomial scan does not allocate pair tensors).

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (``latent_dim=24``, ``degree=3``, B=256) the per-step
overhead of the TMPP head is small compared to the trunk: the
coefficient scan touches a `(B, 3, 24)` tensor 64 times. Throughput
should be within +10% of the i193 baseline.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: first_order_only
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
The diagnostic columns ``primitive_gate``, ``primitive_delta``,
``tmpp_coeff_norm``, and the per-degree coefficient norms
``tmpp_coeff_e2`` / ``tmpp_coeff_e3`` should be inspected to confirm
the gate fires preferentially on positions with multi-piece coalition
structure (e.g. higher e_2 / e_3 norms on tactically active boards).
