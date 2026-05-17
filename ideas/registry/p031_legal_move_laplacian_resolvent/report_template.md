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
- Primitive: LM-LPP (Legal-Move Laplacian Resolvent)
- `lmlpp_alpha` distribution (effective alpha after the `tanh` envelope)
- `lmlpp_mean_feature_norm` / `lmlpp_max_feature_norm`
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-degree positions (`lmlpp_degree_mean` > median)
  - low-degree / endgame positions
- `primitive_delta` distribution on the same two buckets
- Correlation: `primitive_gate` vs `lmlpp_degree_mean`
- Correlation: `primitive_delta` vs final correctness

## Slice Findings

- Target slice: multi-hop tactical / hard-negative near-puzzle positions
  - Required: p031 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`k1_gat_rebrand`) loses >= 50% of that lift
  - Required: A3 (`shuffle_adjacency`) loses >= 70% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Difficulty breakdown: report PR AUC and `primitive_gate` mean per
  `crtk_difficulty` bucket (easy / medium / hard). The Neumann expansion
  is justified only if lift concentrates on medium/hard buckets where
  multi-hop influence matters; the easy bucket must not regress.
- Phase breakdown: report PR AUC and `primitive_gate` mean per
  `crtk_phase` bucket (opening / middlegame / endgame). Lift should
  hold on middlegame (dense legal-move graph) without regressing the
  opening bucket.
- Near-puzzle FP rate at matched recall, broken down by
  `crtk_difficulty` and `crtk_phase`.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | mean alpha | gate mean |
|---|---|---|---|---|
| `none` | | | | |
| `k1_gat_rebrand` | | | | |
| `uniform_piece_weights` | | | | |
| `shuffle_adjacency` | | | | |
| `zero_alpha` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`k1_gat_rebrand`) loses >= 50% of the lift
- [ ] A3 (`shuffle_adjacency`) loses >= 70% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p031.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with sparse CSR):
