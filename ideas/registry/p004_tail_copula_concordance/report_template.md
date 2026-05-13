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
- Primitive: TCC (rank-copula upper-tail concordance)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - true puzzles (fine label 2)
  - non-puzzles (fine labels 0 and 1)
- `tcc_tail_mean`, `tcc_tail_max`, `tcc_channel_mass_max`,
  `tcc_site_mass_max` distributions by class

## Slice Findings

- Target slice: near-puzzle false-positive rate at recall 0.80
  - Required: p004 unablated improves by >= 2% over `i095`-style
    baselines on the same parent
  - Required: A1 (`rank_quantile_only`) loses >= 50% of that lift
- Watch slice: hard / very-hard puzzles
- Watch slice: `crtk_eval_bucket = equal`

## Ablation Comparison Table

| Ablation | near-FP @ recall 0.80 | aggregate PR AUC | mean tail concordance | mean site mass |
|---|---|---|---|---|
| `none` | | | | |
| `rank_quantile_only` | | | | |
| `square_shuffle` | | | | |
| `channel_shuffle` | | | | |
| `single_channel` | | | | |
| `disable_gate` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] near-puzzle FP at recall 0.80 improves by >= 2%
- [ ] A1 (`rank_quantile_only`) loses >= 50% of the near-FP lift
- [ ] A2 (`square_shuffle`) loses >= 70% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 10%

If any box fails: drop p004.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / temperature sweep):
