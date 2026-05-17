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
- Primitive: WCQ (nested adversarial quantifier)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - true puzzles (fine label 2)
  - non-puzzles (fine labels 0 and 1)
- `wcq_value`, `wcq_max_margin`, `wcq_counter_envelope_max`,
  `wcq_witness_entropy` distributions by class

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted WCQ run
must report aggregate metrics plus the fine-label diagnostic matrix,
`slice_report_val.md`, `slice_report_test.md`, and performance by
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`, per-slice false
positives for fine label `1` and false negatives for fine label `2`,
and a short keep/drop conclusion.

## Slice Findings

- Target slice: matched-recall near-puzzle FP at recall 0.80
  - Required: p005 unablated improves by >= 5% over i193 and i011
- Target slice: promotion / underpromotion near-FP
- Target slice: mate-in-1 near-FP
- Watch slice: `crtk_eval_bucket = equal`
- Required per-slice breakouts: `crtk_difficulty` (easy / medium /
  hard) and `crtk_phase` (opening / middlegame / endgame). WCQ must
  not regress any difficulty or phase bucket by more than 0.01 PR AUC
  versus i193.

## Ablation Comparison Table

| Ablation | near-FP @ recall 0.80 | aggregate PR AUC | mean wcq_value | mean witness entropy |
|---|---|---|---|---|
| `none` | | | | |
| `max_claim_only` | | | | |
| `mean_counter_penalty` | | | | |
| `random_counter_assign` | | | | |
| `no_counter_branch` | | | | |
| `disable_gate` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] near-puzzle FP at recall 0.80 improves by >= 5% over i193/i011
- [ ] A1 (`max_claim_only`) loses >= 70% of the near-FP lift
- [ ] A2 (`mean_counter_penalty`) loses >= 50% of the lift
- [ ] promotion / underpromotion near-FP improves over baselines
- [ ] mate-in-1 near-FP improves over baselines
- [ ] `crtk_eval_bucket = equal` slice did not regress

If any box fails: drop p005.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / anneal temperatures /
  combine with PAFR):
