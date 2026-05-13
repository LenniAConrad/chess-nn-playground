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
- Primitive: IDL (incremental delta-linear accumulator)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-material positions
  - low-material / endgame positions
- `primitive_delta` distribution on the same two buckets
- Correlation: `primitive_gate` vs `idl_active_cells`
- Correlation: `primitive_delta` vs final correctness

## Slice Findings

- Target slice: material / pawn-structure heavy positions
  - Required: p025 unablated >= i193 + 0.02 PR AUC
  - Required: A1 (`shuffle_squares`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` — must not regress
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean on positives | gate mean on quiet |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_squares` | | | | |
| `permute_piece_types` | | | | |
| `zero_accumulator` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Target slice lift >= +0.02
- [ ] A1 (`shuffle_squares`) loses >= 70% of the target slice lift
- [ ] A2 (`permute_piece_types`) loses >= 50% of the target slice lift
- [ ] Throughput drop versus i193 < 10% (this head is cheap)

If any box fails: drop p025.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / continue as part of a hybrid):
