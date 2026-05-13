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
- Primitive: PAFR (Pareto antichain frontier over learned utility table)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - true puzzles (fine label 2)
  - non-puzzles (fine labels 0 and 1)
- `primitive_delta` distribution on the same two buckets
- `pafr_frontier_width` mean by class (expect narrower frontiers on
  true puzzles, wider on near-puzzles)
- `pafr_frontier_entropy` mean by class

## Slice Findings

- Target slice: near-puzzle false-positive rate at recall 0.80
  - Required: p001 unablated improves by >= 3% over i193
  - Required: A1 (`shuffle_channels`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` — must not regress
- Watch slice: promotion / underpromotion near-FP versus baselines

## Ablation Comparison Table

| Ablation | near-FP @ recall 0.80 | aggregate PR AUC | mean frontier width | mean frontier entropy |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_channels` | | | | |
| `single_channel` | | | | |
| `scalar_max` | | | | |
| `uniform_frontier` | | | | |
| `disable_gate` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] near-puzzle FP at recall 0.80 improves by >= 3%
- [ ] A1 (`shuffle_channels`) loses >= 70% of the near-FP lift
- [ ] A2 (`single_channel`) does not match the unablated run
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 10%

If any box fails: drop p001.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with different K / C):
