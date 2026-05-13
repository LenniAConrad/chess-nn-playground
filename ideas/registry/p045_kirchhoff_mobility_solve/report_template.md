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

- Mechanism family: `resolvent_pool`
- Primitive: KMS (Kirchhoff Mobility Solve)
- ``kms_potential_norm`` distribution
- ``kms_conductance_mean`` distribution (input-dependent metric should not collapse)
- ``kms_source_norm`` distribution
- `primitive_gate` mean / max / fraction > 0.5 on:
  - low-conductance positions (bottleneck-rich, e.g. fortresses)
  - high-conductance positions (open boards)
- `primitive_delta` distribution on the same buckets

## Slice Findings

- Target slice: "king-safety / fortress / bottleneck" puzzles
  - Required: p045 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`uniform_conductance`) loses >= 35% of that lift
  - Required: A2 (`diagonal_only`) loses >= 30% of that lift

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | conductance_mean / potential_norm |
|---|---|---|---|---|
| `none` | | | | |
| `uniform_conductance` | | | | |
| `diagonal_only` | | | | |
| `shuffle_conductance` | | | | |
| `zero_source` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`uniform_conductance`) loses >= 35% of the lift
- [ ] A2 (`diagonal_only`) loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p045.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):
