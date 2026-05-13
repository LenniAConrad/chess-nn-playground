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
- Primitive: TSDP (terminal-state detection)
- TSDP feature source: in-forward python-chess fallback OR precomputed
  parquet column (record which)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - mate-in-1 positives (`tsdp_mate_in_1 == 1`)
  - non-mate positions (`tsdp_mate_in_1 == 0`)
- `primitive_delta` distribution on the same two buckets
- Correlation: `primitive_gate` vs `tsdp_forcing_density`
- Correlation: `primitive_delta` vs final correctness

## Slice Findings

- Target slice: `crtk_tactic_motifs = mate_in_1`
  - Required: i248 unablated >= i193 + 0.04 PR AUC
  - Required: A1 (`shuffle_tsdp`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` — must not regress
- Watch slice: stalemate-related (`stalemate_threat` indicator firing)
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | mate_in_1 PR AUC | aggregate PR AUC | gate mean on mate positives | gate mean on quiet |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_tsdp` | | | | |
| `disable_gate` | | | | |
| `zero_delta` | | | | |
| `zero_features` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] mate_in_1 slice lift >= +0.04
- [ ] A1 (`shuffle_tsdp`) loses >= 70% of the mate_in_1 lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 25% (or precompute path engaged)

If any box fails: drop i248.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with precompute parquet):
