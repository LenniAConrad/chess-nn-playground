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
- Primitive: RCC (Blahut-Arimoto channel capacity)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - true puzzles (fine label 2)
  - non-puzzles (fine labels 0 and 1)
- `rcc_capacity_nats` and `rcc_capacity_gap` distributions by class
- `rcc_conditional_entropy` and `rcc_output_entropy` distributions

## Slice Findings

- Target slice: near-puzzle false-positive rate at recall 0.80
  - Required: p003 unablated improves by >= 2% over entropy-only
    baselines on the same parent
  - Required: A1 (`entropy_only`) loses >= 50% of that lift
- Watch slice: `crtk_eval_bucket = equal`
- Watch slice: promotion / underpromotion near-FP

Per `ideas/docs/BENCHMARK_REPORTING.md`, also report slice tables
sliced by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The slice tables live
in `slice_report_val.md` and `slice_report_test.md` and must include:

- per-slice FP rate at the matched recall threshold for fine label `1`
- per-slice FN rate for fine label `2`
- per-slice mean `primitive_gate`, `rcc_capacity_nats`, and
  `rcc_capacity_gap`
- per-`crtk_difficulty` and per-`crtk_phase` confidence/calibration

The keep/drop decision below is read together with the
`crtk_difficulty` and `crtk_phase` breakdowns: a primitive that only
lifts the easy bucket and regresses on hard / very-hard or in the
endgame `crtk_phase` is dropped.

## Ablation Comparison Table

| Ablation | near-FP @ recall 0.80 | aggregate PR AUC | mean capacity | mean capacity gap |
|---|---|---|---|---|
| `none` | | | | |
| `entropy_only` | | | | |
| `row_shuffle_channel` | | | | |
| `duplicate_rows` | | | | |
| `uniform_replies` | | | | |
| `disable_gate` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] near-puzzle FP at recall 0.80 improves by >= 2%
- [ ] A1 (`entropy_only`) loses >= 50% of the near-FP lift
- [ ] A2/A3/A4 (capacity killers) lose >= 70% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 10%

If any box fails: drop p003.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with different K / R / T):
