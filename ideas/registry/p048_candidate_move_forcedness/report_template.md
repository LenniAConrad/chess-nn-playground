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

- Mechanism family: `move_graph`
- Primitive: CMF (Candidate Move Forcedness)
- `cmf_top1_score` distribution
- `cmf_gap12` distribution
- `cmf_topk_mass` distribution
- `cmf_entropy` distribution
- `cmf_move_count` distribution
- `cmf_check_peak`, `cmf_capture_peak`, `cmf_promotion_peak`,
  `cmf_see_peak` distributions
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-`cmf_top1_score` positions
  - low-entropy positions (singular forcing candidate)
- `primitive_delta` distribution on the same two buckets

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted idea
must report aggregate metrics plus the fine-label diagnostic
matrix, `slice_report_val.md`, `slice_report_test.md`, and
performance broken down by `crtk_difficulty`, `crtk_phase`,
`crtk_eval_bucket`, `crtk_tactic_motifs`, and
`crtk_tag_families`. Include per-slice false positives for fine
label `1`, per-slice false negatives for fine label `2`,
confidence / calibration by slice, and the highest-confidence
wrong examples with FEN, difficulty, phase, and motifs.

## Slice Findings

- Target slice: "forcing-line tactics" puzzles
  (mate_in_1, hanging, fork, overload, promotion, discovered_attack)
  - Required: p048 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`deterministic_score`) loses >= 50% of that lift
  - Required: A2 (`mean_pool`) loses >= 30% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Required `crtk_difficulty` breakdown: lift must concentrate on
  medium / hard / very_hard buckets without regressing the easy
  bucket.
- Required `crtk_phase` breakdown: lift must hold on middlegame and
  endgame buckets; opening-bucket non-regression watched.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | mean top1_score |
|---|---|---|---|---|
| `none` | | | | |
| `deterministic_score` | | | | |
| `mean_pool` | | | | |
| `flags_only` | | | | |
| `dense_edges` | | | | |
| `no_consequence` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`deterministic_score`) loses >= 50% of the lift
- [ ] A2 (`mean_pool`) loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 20%

If any box fails: drop p048.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):
