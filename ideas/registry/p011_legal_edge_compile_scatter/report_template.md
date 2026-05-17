# Report Template — p011 Legal-Edge Compile Scatter

## Run

- Result path:
- Config: `ideas/registry/p011_legal_edge_compile_scatter/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy / F1 / ROC AUC / PR AUC / Calibration:

## Architecture-Specific Diagnostics

- Mechanism family: `legal_routing`
- Primitive: Legal-Edge Compile Scatter (typed σ-gated message
  scatter on a content-compiled adjacency).
- `lecs_gate_mean` distribution: mean σ-gate over valid edges; values
  near 0 or 1 across the entire validation set indicate gate
  degeneration (one of the failure modes flagged in `ablations.md`).
- `primitive_gate` mean / max / fraction > 0.5, and `primitive_delta`
  distribution, on:
  - Positions with at least one slider
  - Positions without sliders

## Slice Findings

- Declared target slice: positions where typed legal edges have
  heterogeneous importance (overloaded defenders, attacker-defender
  swap chains).
  - Required: p011 unablated >= i193 + 0.03 PR AUC on slice
  - Required: `no_edge_gate` ablation loses >= 50% of that lift
- Performance broken down by `crtk_difficulty`, `crtk_phase`,
  `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`
  per `ideas/docs/BENCHMARK_REPORTING.md`.
- Watch slice: aggregate FP rate at matched recall; `crtk_eval_bucket = equal`
  must not regress.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean | knight msg norm |
|---|---|---|---|---|
| `none` | | | | |
| `no_edge_gate` | | | | |
| `random_typed_edges` | | | | |
| `shared_type_weight` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.03
- [ ] `no_edge_gate` ablation loses >= 50% of the lift
- [ ] `random_typed_edges` ablation loses >= 70% of the lift
- [ ] Throughput drop versus i193 < 30%

If any box fails: drop p011.
