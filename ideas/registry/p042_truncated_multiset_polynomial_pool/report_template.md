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

- Mechanism family: `ledger_polynomial`
- Primitive: TMPP (Truncated Multiset Polynomial Pool)
- ``tmpp_active_mean`` distribution
- ``tmpp_coeff_norm`` distribution
- ``tmpp_coeff_e1`` / ``tmpp_coeff_e2`` / ``tmpp_coeff_e3`` distributions
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high-coalition-norm positions (``tmpp_coeff_e2`` > median)
  - sparse-piece / endgame positions (``tmpp_active_mean`` < median)
- `primitive_delta` distribution on the same two buckets

## Slice Findings

- Target slice: "multi-piece coalition / overloaded defender" tactical positions
  - Required: p042 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`first_order_only`) loses >= 50% of that lift
  - Required: A2 (`uniform_mask`) loses >= 30% of that lift
- Per-slice breakdowns required (must not regress vs i193):
  - `crtk_difficulty` buckets (easy / medium / hard) — multi-piece
    polynomial coalitions are expected to lift the medium / hard
    buckets where overloaded defenders dominate, without regressing
    the easy bucket.
  - `crtk_phase` buckets (opening / middlegame / endgame) — lift
    should concentrate on middlegame positions where coalition-rich
    structures survive truncation at `K = 3`, with no opening-bucket
    regression.
  - `crtk_eval_bucket`, `crtk_tactic_motifs`, `crtk_tag_families`.
- Per-slice false-positive rate for fine label `1` and false-negative
  rate for fine label `2`, jointly stratified by `crtk_difficulty` x
  `crtk_phase`.
- Highest-confidence wrong examples must report FEN,
  `crtk_difficulty`, `crtk_phase`, and motifs.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | tmpp_coeff_e2 mean |
|---|---|---|---|---|
| `none` | | | | |
| `first_order_only` | | | | |
| `uniform_mask` | | | | |
| `shuffle_mask` | | | | |
| `shuffle_tokens` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`first_order_only`) loses >= 50% of the lift
- [ ] A2 (`uniform_mask`) loses >= 30% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 15%

If any box fails: drop p042 (or upgrade to a fused polynomial-scan kernel).

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):
