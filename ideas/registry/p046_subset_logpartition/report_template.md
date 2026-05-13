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

- Mechanism family: `log_semiring_ledger`
- Primitive: SLPT (Bounded Subset Log-Partition Transform)
- ``slpt_active_mean`` distribution
- ``slpt_logpartition_norm`` distribution
- ``slpt_y1`` / ``slpt_y2`` / ``slpt_y3`` distributions
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-coalition positions (``slpt_y2`` > median)
  - sparse-piece / endgame positions (``slpt_active_mean`` < median)
- `primitive_delta` distribution on the same two buckets

## Slice Findings

- Target slice: "k-defenders threshold / multi-attacker race" puzzles
  - Required: p046 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`k1_only`) loses >= 50% of that lift
  - Required: A2 (`uniform_mask`) loses >= 30% of that lift

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | slpt_y2 mean |
|---|---|---|---|---|
| `none` | | | | |
| `k1_only` | | | | |
| `uniform_mask` | | | | |
| `shuffle_mask` | | | | |
| `shuffle_tokens` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`k1_only`) loses >= 50% of the lift
- [ ] A2 (`uniform_mask`) loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p046.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):
