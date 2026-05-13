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
- Primitive: RayPool (ray-cast obstacle pooling)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - long-range tactical slices
  - quiet positions
- `primitive_delta` distribution on the same two buckets
- Correlation: `primitive_gate` vs `raypool_max_dir_energy`
- Learned per-direction `gamma` values (post-train)

## Slice Findings

- Target slice: long-range tactical motifs (back-rank, skewer, battery,
  pin)
  - Required: p026 unablated >= i193 + 0.02 PR AUC
  - Required: A1 (`drop_occlusion`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` — must not regress

## Ablation Comparison Table

| Ablation | long-range slice PR AUC | aggregate PR AUC | gate mean | gamma mean |
|---|---|---|---|---|
| `none` | | | | |
| `drop_occlusion` | | | | |
| `shuffle_directions` | | | | |
| `zero_rays` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Long-range slice lift >= +0.02
- [ ] A1 (`drop_occlusion`) loses >= 70% of the long-range slice lift
- [ ] A2 (`shuffle_directions`) loses >= 50% of the long-range lift
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p026.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / continue as part of a hybrid):
