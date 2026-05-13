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
- Primitive: SLMGT (Sparse Legal-Move Graph Transition)
- ``slmgt_degree_mean`` distribution
- ``slmgt_edge_norm`` / ``slmgt_edge_max`` distributions
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-piece-density positions (``slmgt_degree_mean`` > median)
  - low-piece-density / endgame positions
- `primitive_delta` distribution on the same two buckets

## Slice Findings

- Target slice: "hanging piece" / pin / fork tactical positions
  - Required: p035 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`separable_phi`) loses >= 40% of that lift
  - Required: A2 (`uniform_adjacency`) loses >= 30% of that lift

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | edge_norm mean |
|---|---|---|---|---|
| `none` | | | | |
| `separable_phi` | | | | |
| `uniform_adjacency` | | | | |
| `shuffle_adjacency` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`separable_phi`) loses >= 40% of the lift
- [ ] A2 (`uniform_adjacency`) loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 30% (or plan sparse-edge upgrade)

If any box fails: drop p035 (or upgrade to sparse-edge kernel).

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):
