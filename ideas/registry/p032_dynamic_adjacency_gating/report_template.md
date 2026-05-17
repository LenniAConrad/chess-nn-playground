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

- Mechanism family: `legal_graph`
- Primitive: DAG (Dynamic Adjacency Gating)
- `dag_total_degree` distribution
- Per-type ``dag_degree_type_<code>`` shares -- which move type dominates
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-degree positions (`dag_total_degree` > median)
  - low-degree / endgame positions
- `primitive_delta` distribution on the same two buckets
- Correlation: `primitive_gate` vs `dag_total_degree`

## Slice Findings

- Target slice: move-type-specialised positions (open files,
  knight-outpost tactics, long-diagonal pins)
  - Required: p032 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`single_move_type`) loses >= 50% of that lift
  - Required: A4 (`shuffle_adjacency`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Difficulty breakdown: report PR AUC and `primitive_gate` mean per
  `crtk_difficulty` bucket (easy / medium / hard). The move-type-aware
  routing is justified only if lift concentrates on medium/hard buckets
  where adjacency specialisation matters; the easy bucket must not regress.
- Phase breakdown: report PR AUC and `primitive_gate` mean per
  `crtk_phase` bucket (opening / middlegame / endgame). Lift should hold
  on middlegame (dense, varied legal-move topology) without regressing
  endgame positions where adjacency collapses.
- Near-puzzle FP rate at matched recall, broken down by
  `crtk_difficulty` and `crtk_phase`.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | total_degree mean |
|---|---|---|---|---|
| `none` | | | | |
| `single_move_type` | | | | |
| `soft_mask` | | | | |
| `uniform_adjacency` | | | | |
| `shuffle_adjacency` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`single_move_type`) loses >= 50% of the lift
- [ ] A4 (`shuffle_adjacency`) loses >= 70% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p032.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):
