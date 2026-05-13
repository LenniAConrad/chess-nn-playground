# Report Template

## Run

- Result path:
- Config:
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Architecture-Specific Diagnostics

- Mechanism family: `king_path` (inherits i193 framing)
- Primitive: Ray-SSM (ray-parallel selective state space)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - long-range mixing positives
  - quiet positions
- `primitive_delta` distribution on the same two buckets
- Distributions of `ray_ssm_mean_A` and `ray_ssm_mean_B`

## Slice Findings

- Target slice: long-range mixing / piece-value-along-ray positions
  - Required: p030 unablated >= i193 + 0.02 PR AUC
  - Required: A1 (`disable_selective_A`) or A2 (`disable_selective_B`)
    loses >= 70% of that lift
  - Cross-check: unablated p030 > p026 RayPool and > p029 OARS on the
    same slice
- Watch slice: `crtk_eval_bucket = equal` — must not regress

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | mean_A | mean_B |
|---|---|---|---|---|---|
| `none` | | | | | |
| `disable_selective_A` | | | | | |
| `disable_selective_B` | | | | | |
| `no_directional_C` | | | | | |
| `zero_ssm_features` | | | | | |
| `zero_delta` | | | | | |
| `disable_gate` | | | | | |
| `trunk_only` | | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Target slice lift >= +0.02
- [ ] A1 or A2 loses >= 70% of the target slice lift
- [ ] Unablated p030 beats p026 and p029 on the target slice
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p030.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / continue as part of a hybrid):
