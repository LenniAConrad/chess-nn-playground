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

- Mechanism family: `rejection_veto`
- Primitive: NPHN (Near-Puzzle Hard-Negative Veto)
- `nphn_veto_pressure` distribution (true puzzle vs near-puzzle)
- `nphn_forcedness_gap` distribution
- `nphn_forcedness_at_mstar` distribution
- `nphn_legality_discount` distribution
- `nphn_candidate_concentration` distribution
- `nphn_reply_availability` distribution
- `nphn_reply_channel_information` distribution
- `primitive_gate` mean / max / fraction > 0.5 on:
  - near-puzzle slice (`crtk_eval_bucket = equal` and similar)
  - true-puzzle slice
- `primitive_delta` distribution on the same two slices (must be
  non-positive everywhere by construction)

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted idea must
report aggregate metrics plus the fine-label diagnostic matrix,
`slice_report_val.md`, `slice_report_test.md`, and performance broken
down by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. Include per-slice false
positives for fine label `1`, per-slice false negatives for fine label
`2`, confidence/calibration by slice, and the highest-confidence wrong
examples with FEN, difficulty, phase, and motifs.

The operational metric for p055 is the **matched-recall near-puzzle FP
rate**: validation-derived thresholds `tau_R` for `R in {0.80, 0.85}`
applied at test time. Report:

- `NearFP@0.80`, `NearFP@0.85`
- `FarFP@0.80`, `FarFP@0.85`
- `TotalFP@0.80`, `TotalFP@0.85`
- `Precision@0.80`, `Precision@0.85`
- `delta NearFP@R` vs i193 parent

## Slice Findings

- Target slice: near-puzzle FP rate at recall `{0.80, 0.85}`
  - Required: p055 unablated reduces near-puzzle FP at recall 0.80 by
    at least 3% relative vs i193
  - Required: p055 unablated reduces near-puzzle FP at recall 0.85 by
    at least 3% relative vs i193
  - Required: A1 (`no_replies`) loses >= 50% of that lift
  - Required: A2 (`no_legality_discount`) loses >= 50% of that lift
  - Required: A3 (`concentration_only`) loses >= 30% of that lift
- Watch slices: `crtk_eval_bucket = equal`, `crtk_difficulty = hard`,
  `crtk_difficulty = very_hard`, mate-in-1 near-puzzles,
  promotion / underpromotion slices -- must not regress more than the
  aggregate threshold.

## Ablation Comparison Table

| Ablation | NearFP@0.80 | NearFP@0.85 | aggregate PR AUC | gate mean | nphn_veto_pressure mean |
|---|---|---|---|---|---|
| `none` | | | | | |
| `no_replies` | | | | | |
| `no_legality_discount` | | | | | |
| `concentration_only` | | | | | |
| `shuffle_replies` | | | | | |
| `no_overload` | | | | | |
| `no_king_escape` | | | | | |
| `zero_delta` | | | | | |
| `trunk_only` | | | | | |
| `disable_gate` | | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] NearFP@0.80 relative reduction >= 3% vs i193
- [ ] NearFP@0.85 relative reduction >= 3% vs i193
- [ ] A1 (`no_replies`) loses >= 50% of the lift
- [ ] A2 (`no_legality_discount`) loses >= 50% of the lift
- [ ] A3 (`concentration_only`) loses >= 30% of the lift
- [ ] Watch slices (equal/hard/very_hard/mate1/promotion) did not regress
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p055.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):
