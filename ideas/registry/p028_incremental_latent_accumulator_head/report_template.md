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
- Primitive: ILA (incremental latent accumulator)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - king-safety / king-zone positives
  - quiet positions
- `primitive_delta` distribution on the same two buckets
- Correlation: `primitive_gate` vs `ila_king_norm`
- Correlation: `primitive_delta` vs final correctness

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted idea must
report aggregate metrics plus the fine-label diagnostic matrix,
`slice_report_val.md`, `slice_report_test.md`, and performance broken
down by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. Include per-slice false
positives for fine label `1`, per-slice false negatives for fine label
`2`, confidence/calibration by slice, and the highest-confidence wrong
examples with FEN, difficulty, phase, and motifs.

## Slice Findings

- Target slice: king-safety / king-zone / endgame positions
  - Required: p028 unablated >= i193 + 0.02 PR AUC
  - Required: A1 (`zero_king_accumulator`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` — must not regress
- Required `crtk_difficulty` breakdown: lift must concentrate on
  medium/hard buckets (where king-context evaluation matters most)
  without regressing the easy bucket.
- Required `crtk_phase` breakdown: lift must hold on middlegame and
  endgame buckets (king-safety and endgame king activity), with no
  opening-bucket regression.
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | king_norm mean |
|---|---|---|---|---|
| `none` | | | | |
| `zero_king_accumulator` | | | | |
| `zero_global_accumulator` | | | | |
| `linear_only` | | | | |
| `shuffle_square_order` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Target slice lift >= +0.02
- [ ] A1 (`zero_king_accumulator`) loses >= 70% of the king-related lift
- [ ] A3 (`linear_only`) shows meaningful lift over p025 IDL (otherwise drop p028 in favour of p025)
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p028.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / continue as part of a hybrid):
