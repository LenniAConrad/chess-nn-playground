# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = efficient_ray_occlusion_scan`
- `model.head_hidden_dim`, `model.head_dropout`, `model.gate_init`,
  `model.ablation` for the EROS head.
- Trunk hyperparameters retain their i193 names with a `trunk_` prefix.
- `training.batch_size` stays at the i193 default 256; the scan only
  allocates `(B, 8, 64, 7, 16)` ray feature tensors plus the four
  `(B, 8, 64, 7)` mask tensors, so memory cost is similar to p020.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (B=256) the per-step overhead of the EROS head is small
compared to the trunk: one `gather` over a `(B, 8, 64, 7, 16)` tensor,
one `cumsum` over a `(B, 8, 64, 7)` tensor, and a handful of pointwise
ops. The delta and gate MLPs are small (head_hidden_dim=64). Expected
throughput within +10% of the i193 baseline; verify with the benchmark
script described in `implementation_notes.md` once it is added.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: first_only
```

Additional ablations:

- `model.ablation: no_blocker_id`     -- side / value identity falsifier
- `model.ablation: uniform_occupancy` -- mask-irrelevance control
- `model.ablation: empty_occupancy`   -- pure-geometry control
- `model.ablation: shuffle_occupancy` -- decouples mask from position
- `model.ablation: zero_delta`        -- i193 baseline
- `model.ablation: trunk_only`        -- strongest control
- `model.ablation: disable_gate`      -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The `eros_first_blocker_rate` / `eros_second_blocker_rate` columns
should differ between sparse-piece endgames and crowded middlegames;
otherwise the geometry signal is not informative. The
`primitive_gate` should not collapse to 0 across the board; if the
`zero_delta` ablation matches the unablated head in aggregate, the
head is dead weight.
