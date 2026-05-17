# Report Template — p010 Ray-Occlusion Semiring Scan

## Run

- Result path:
- Config: `ideas/registry/p010_ray_occlusion_semiring_scan/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy / F1 / ROC AUC / PR AUC / Calibration:

## Architecture-Specific Diagnostics

- Mechanism family: `ray_attention`
- Primitive: Ray-Occlusion Semiring Scan
- `ros_mean_transmittance` distribution (lower for crowded boards,
  higher for sparse end-games).
- `ros_step_decay_mean` per direction — collapse to {0, 1} indicates
  the decay parameter has degenerated.
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positions with at least one slider
  - Positions without sliders
- `primitive_delta` distribution on the same two buckets

## Slice Findings

- Declared target slice: `crtk_tactic_motifs in {pin, skewer, x_ray, discovered_attack}`.
  - Required: p010 unablated >= i193 + 0.04 PR AUC on slice
  - Required: `uniform_transmittance` ablation loses >= 70% of that lift
- Performance broken down by `crtk_difficulty`, `crtk_phase`,
  `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`
  per `ideas/docs/BENCHMARK_REPORTING.md`.
- Watch slice: aggregate FP rate at matched recall; `crtk_eval_bucket = equal`
  must not regress.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | mean transmittance | mean step decay |
|---|---|---|---|---|
| `none` | | | | |
| `uniform_transmittance` | | | | |
| `constant_direction` | | | | |
| `no_step_decay` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.04
- [ ] `uniform_transmittance` ablation loses >= 70% of the lift
- [ ] `constant_direction` ablation loses >= 30% of the lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p010.
