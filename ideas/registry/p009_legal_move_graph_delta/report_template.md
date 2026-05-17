# Report Template — p009 Legal-Move-Graph Convolution

## Run

- Result path:
- Config: `ideas/registry/p009_legal_move_graph_delta/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted idea must
require aggregate metrics plus the fine-label diagnostic matrix,
`slice_report_val.md`, `slice_report_test.md`, performance by
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`, per-slice false
positives for fine label `1` and false negatives for fine label `2`,
confidence/calibration by slice, the highest-confidence wrong examples
(FEN, `crtk_difficulty`, `crtk_phase`, motifs), and a short
keep/drop conclusion.

## Aggregate Metrics

- Accuracy / F1 / ROC AUC / PR AUC / Calibration:

## Architecture-Specific Diagnostics

- Mechanism family: `legal_routing`
- Primitive: LMGConv (typed legal-move-graph convolution)
- Per-type message norms (`lmgconv_msg_norm_{P,N,B,R,Q,K}`):
  distribution across the validation set; should reflect per-piece-
  type prevalence on the board.
- `lmgconv_edge_count` distribution vs `primitive_gate`.
- `primitive_gate` mean/max on tactical positives vs quiet positions,
  broken out by `crtk_difficulty` and `crtk_phase`.

## Slice Findings

- Target slice:
  `crtk_tactic_motifs in {knight_fork, rook_battery, bishop_pin}`,
  broken out by `crtk_difficulty` and `crtk_phase` so we can tell
  whether lift is concentrated on easy puzzles or on a single game
  phase.
- Watch slice: aggregate FP rate at matched recall.
- Watch slice: `crtk_eval_bucket = equal` — must not regress.
- Performance must be broken out by `crtk_difficulty` and `crtk_phase`
  buckets.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | knight msg norm | rook msg norm |
|---|---|---|---|---|
| `none` | | | | |
| `random_typed_edges` | | | | |
| `shared_weight` | | | | |
| `no_normalization` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.03
- [ ] `random_typed_edges` ablation loses >= 70% of the lift
- [ ] `shared_weight` ablation loses >= 50% of the lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p009.

## Conclusions

- What the operator appears able to learn:
- What it appears unable to learn:
- Highest-confidence wrong examples (FEN, `crtk_difficulty`,
  `crtk_phase`, motifs):
- Recommended next step:
