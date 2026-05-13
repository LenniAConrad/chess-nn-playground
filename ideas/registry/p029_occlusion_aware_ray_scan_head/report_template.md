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
- Primitive: OARS (occlusion-aware ray scan)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - x-ray / pin / battery positives
  - quiet positions
- `primitive_delta` distribution on the same two buckets
- Distribution of `oars_mean_blocker_gate` (post-train)

## Slice Findings

- Target slice: x-ray / pin / battery positions
  - Required: p029 unablated >= i193 + 0.02 PR AUC
  - Required: A1 (`disable_blocker_gate`) loses >= 70% of that lift
  - Cross-check: unablated p029 > p026 RayPool on the same slice
- Watch slice: `crtk_eval_bucket = equal` — must not regress

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | blocker_gate mean |
|---|---|---|---|---|
| `none` | | | | |
| `disable_blocker_gate` | | | | |
| `shuffle_directions` | | | | |
| `zero_oars_features` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Target slice lift >= +0.02
- [ ] A1 (`disable_blocker_gate`) loses >= 70% of the target lift
- [ ] Unablated p029 beats p026 RayPool on the target slice
- [ ] Throughput drop versus i193 < 20%

If any box fails: drop p029.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / continue as part of a hybrid):
