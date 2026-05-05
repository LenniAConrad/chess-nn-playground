# Idea Report Template

## Required Comparisons

- size-matched CNN trunk
- LC0 BT4 benchmark
- `no_flow_solver`
- `shuffled_compatibility`
- `no_capacity_constraints`
- king-only and material-only obligation subsets

## Required Metrics

- test F1
- test PR AUC
- test accuracy
- 3x2 source-class confusion matrix plus `slice_report_val.md` and `slice_report_test.md`
- random -> puzzle false-positive rate
- near-puzzle -> puzzle false-positive rate
- puzzle recall
- mean residual by source class
- mean residual by obligation type

## Required Slice Analysis

Follow `ideas/BENCHMARK_REPORTING.md`. In this idea's report, add:

- flow residual, accuracy, and false-positive rate by `crtk_difficulty`;
- flow residual by `crtk_phase`, especially endgame and promotion-like positions;
- motif slices for `overload`, `pin`, `skewer`, `mate_in_1`, `hanging`, `fork`, and `(none)`;
- obligation-type residuals inside the best and worst motifs;
- examples where high residual correctly vetoed a near-puzzle false positive and examples where it failed.

The obligation-flow claim is supported only if residuals separate true puzzles from near-puzzles in resource-constrained motifs such as `overload`, `pin`, `skewer`, and `mate_in_1`, especially in `hard`/`very_hard` rows. If residuals mostly track material/eval bucket or do not improve near-puzzle false positives by slice, the solver bottleneck is not justified.

## Decision Rule

Continue only if the flow bottleneck improves the near-puzzle row or gives clearly useful diagnostics that simpler pooling cannot.
