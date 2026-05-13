# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = grassmann_rook_pool`
- `model.num_attackers` / `model.num_defenders` set the bipartite grid
  size (default 8 each).
- `model.token_dim`, `model.score_channels`, `model.degree`,
  `model.head_hidden_dim`, `model.head_dropout`, `model.gate_init`,
  `model.ablation` for the GRMP head.
- trunk hyperparameters retain their i193 names with a `trunk_` prefix.
- ``training.batch_size`` stays at the i193 default 256; the GRMP edge
  tensor is small.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (``num_attackers=num_defenders=8``, ``token_dim=32``,
``score_channels=8``, ``degree=2``, B=256), throughput should be
within +15% of the i193 baseline. K=3 adds an O(R*C) Python loop;
expect another 5-10% slowdown at K=3.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: drop_exclusion
```

Additional ablations:

- `model.ablation: scalar_score`        -- single-channel score control
- `model.ablation: shuffle_attackers`   -- rule-feature falsifier (attacker side)
- `model.ablation: shuffle_defenders`   -- rule-feature falsifier (defender side)
- `model.ablation: zero_delta`          -- i193 baseline
- `model.ablation: trunk_only`          -- strongest control
- `model.ablation: disable_gate`        -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns ``primitive_gate``, ``primitive_delta``,
``grmp_coeff_e2`` (the matching-coefficient channel that distinguishes
this primitive from a bilinear pool) should be inspected to confirm
the gate fires preferentially on positions with overloaded defenders.
