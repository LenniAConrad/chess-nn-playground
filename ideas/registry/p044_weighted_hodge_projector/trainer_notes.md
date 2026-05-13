# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = weighted_hodge_projector`
- `model.flow_channels`, `model.edge_feature_dim`,
  `model.head_hidden_dim`, `model.head_dropout`, `model.gate_init`,
  `model.solve_eps`, `model.ablation` for the WHP head.
- trunk hyperparameters retain their i193 names with a `trunk_` prefix.
- ``training.batch_size`` may be reduced to 128 if the SPD solves push
  memory on smaller GPUs; the default 256 should fit comfortably.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (``flow_channels=4``, ``edge_feature_dim=16``, B=256), each
forward pass runs two batched SPD solves of size 64x64 and 49x49.
Throughput should be within +20% of the i193 baseline. If the
vertex-Laplacian solve dominates, the next upgrade is a precomputed
Cholesky factorisation when the metric is fixed (i.e. the
`uniform_metric` ablation; this is only useful as a cost reference).

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: uniform_metric
```

Additional ablations:

- `model.ablation: drop_curl`         -- circulation component test
- `model.ablation: drop_gradient`     -- gradient component test
- `model.ablation: drop_harmonic`     -- harmonic component test
- `model.ablation: shuffle_edge_flow` -- rule-feature falsifier
- `model.ablation: zero_delta`        -- i193 baseline
- `model.ablation: trunk_only`        -- strongest control
- `model.ablation: disable_gate`      -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostics ``whp_gradient_energy`` / ``whp_curl_energy`` /
``whp_harmonic_energy`` should reflect plausible chess geometry on
hand-picked positions (e.g. fortresses should have low gradient and
non-zero harmonic energy; mating nets should have high gradient
energy directed at the king zone).
