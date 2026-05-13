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
- Primitive: RSP (entropy-regularized saddle over learned payoff table)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - true puzzles (fine label 2)
  - non-puzzles (fine labels 0 and 1)
- `primitive_delta` distribution on the same two buckets
- `rsp_saddle_value` and `rsp_exploitability` distributions by class
- `rsp_attacker_entropy` and `rsp_defender_entropy` distributions

## Slice Findings

- Target slice: near-puzzle false-positive rate at recall 0.80
  - Required: p002 unablated improves by >= 3% over i193
  - Required: A1 (`row_shuffle_payoff`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal`
- Watch slice: hard / very-hard cyclic tactical positions

## Ablation Comparison Table

| Ablation | near-FP @ recall 0.80 | aggregate PR AUC | mean saddle value | mean exploitability |
|---|---|---|---|---|
| `none` | | | | |
| `row_shuffle_payoff` | | | | |
| `col_shuffle_payoff` | | | | |
| `uniform_payoff` | | | | |
| `pure_max_min` | | | | |
| `disable_gate` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] near-puzzle FP at recall 0.80 improves by >= 3%
- [ ] A1 (`row_shuffle_payoff`) loses >= 70% of the near-FP lift
- [ ] A2 (`col_shuffle_payoff`) loses >= 50% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 10%

If any box fails: drop p002.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with different K / R / T):
