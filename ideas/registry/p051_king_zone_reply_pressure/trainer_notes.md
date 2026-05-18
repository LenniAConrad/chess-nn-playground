# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config
mirrors the i193 baseline (same split, encoding, seed, budget,
threshold rule).

Differences vs i193:

- `model.name = king_zone_reply_pressure`
- `model.head_hidden_dim`, `model.head_dropout`, `model.gate_init`,
  `model.ablation` for the KZRP head
- trunk hyperparameters retain their i193 names with a `trunk_` prefix
- ``training.batch_size`` stays at the i193 default 256 (the head
  adds modest memory; the zone-pressure core never materialises a
  `(B, N, N, N)` triple tensor, only `(B, N, N)` einsums shared with
  p050)

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (head_hidden_dim=64, B=256) the per-step overhead of the
KZRP head is small compared to the trunk. The attack builder does a
batched `(B, 64, 64)` einsum per colour (nominal + free, so two per
colour). The side vector then does several `(B, 64)` reductions and
two `(B, 64)` × `(64, 64)` einsums for ring / front projection.
Throughput should be within +10..15% of the i193 baseline.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: no_front_zone
```

Additional ablations:

- `model.ablation: no_pins`              -- pin / fake-defense test
- `model.ablation: uniform_zone_weights` -- zone-weight load-bearing
- `model.ablation: no_escape_decomp`     -- escape decomposition test
- `model.ablation: uniform_units`        -- CPW attack-unit load-bearing
- `model.ablation: no_asymmetry`         -- side-to-move asymmetry test
- `model.ablation: zero_delta`           -- i193 baseline (numeric recovery)
- `model.ablation: trunk_only`           -- strongest control
- `model.ablation: disable_gate`         -- gate load-bearing

## Reports

Standard idea report; required slices listed in
`report_template.md`. The diagnostic columns ``primitive_gate``,
``primitive_delta``, ``kzrp_us_zone_pressure``,
``kzrp_them_zone_pressure``, ``kzrp_us_live_escapes``,
``kzrp_us_king_attack_mass``, and ``kzrp_asym_score`` should be
inspected to confirm the gate fires preferentially on positions with
high king-zone pressure (especially `mate_in_1`-style positions
where the attacker has a forcing king attack). The unablated run
should show non-trivial ``kzrp_us_zone_pressure`` and
``kzrp_us_king_attack_mass`` on the ``mate_in_1`` and
``discovered_attack`` motif slices.
