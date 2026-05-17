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

- Mechanism family: `hodge_projection`
- Primitive: WHP (Weighted Hodge Projector)
- ``whp_gradient_energy`` / ``whp_curl_energy`` / ``whp_harmonic_energy`` distributions
- ``whp_flow_energy`` distribution
- ``whp_weight_mean`` distribution (input-dependent metric should not collapse)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - high curl-energy positions
  - high gradient-energy positions
  - high harmonic-energy positions
- `primitive_delta` distribution on the same buckets

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

- Target slice: "fortress / circulation / trapped pressure" puzzles
  - Required: p044 unablated >= i193 + 0.02 PR AUC on target slice
  - Required: A1 (`uniform_metric`) loses >= 35% of that lift
  - Required: at least one of {A2, A3, A4} loses >= 25% of that lift
- Watch slice: `crtk_eval_bucket = equal` -- must not regress
- Required `crtk_difficulty` breakdown: lift must concentrate on
  medium/hard buckets (fortress / trapped-pressure positions are rare
  in the easy bucket) without regressing the easy bucket.
- Required `crtk_phase` breakdown: lift must hold on middlegame and
  endgame buckets, with no opening-bucket regression.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | gradient/curl/harmonic energy means |
|---|---|---|---|---|
| `none` | | | | |
| `uniform_metric` | | | | |
| `drop_curl` | | | | |
| `drop_gradient` | | | | |
| `drop_harmonic` | | | | |
| `shuffle_edge_flow` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |
| `disable_gate` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.02
- [ ] A1 (`uniform_metric`) loses >= 35% of the lift
- [ ] At least one component-drop ablation loses >= 25% of the lift
- [ ] `crtk_eval_bucket = equal` slice did not regress
- [ ] Throughput drop versus i193 < 20%

If any box fails: drop p044.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop):
