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

- Mechanism family: `rook_matching`
- Primitive: GRMP (Grassmann Rook-Matching Pool)
- ``grmp_attacker_count`` / ``grmp_defender_count`` distributions
- ``grmp_coeff_norm`` distribution
- ``grmp_coeff_e1`` / ``grmp_coeff_e2`` / ``grmp_coeff_e3`` distributions
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high attacker-count positions
  - low attacker-count / endgame positions
- `primitive_delta` distribution on the same two buckets

## Slice Findings

- Target slice: "overloaded defender / two-attacker race" tactical positions
  - Required: p043 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`drop_exclusion`) loses >= 40% of that lift
  - Required: A2 (`scalar_score`) loses >= 25% of that lift

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | grmp_coeff_e2 mean |
|---|---|---|---|---|
| `none` | | | | |
| `drop_exclusion` | | | | |
| `scalar_score` | | | | |
| `shuffle_attackers` | | | | |
| `shuffle_defenders` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`drop_exclusion`) loses >= 40% of the lift
- [ ] A2 (`scalar_score`) loses >= 25% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 20%

If any box fails: drop p043.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):
