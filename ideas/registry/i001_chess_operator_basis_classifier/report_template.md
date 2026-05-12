# Idea Report Template

## Required Comparisons

- simple CNN size-matched
- LC0 BT4 tower
- operator basis without ray operators
- operator basis with shuffled relation masks

## Required Metrics

- test F1
- test PR AUC
- test accuracy
- 3x2 source-class matrix plus `slice_report_val.md` and `slice_report_test.md`
- random -> puzzle false-positive rate
- near-puzzle -> puzzle false-positive rate
- puzzle recall

## Required Slice Analysis

Follow `ideas/docs/BENCHMARK_REPORTING.md`. In this idea's report, add:

- accuracy, recall, false-positive rate, and confidence by `crtk_difficulty`;
- the same metrics by `crtk_phase`, with special attention to endgames where operator bases can overfit material patterns;
- motif rows for `pin`, `skewer`, `discovered_attack`, `fork`, `hanging`, `overload`, and `(none)`;
- delta versus the size-matched CNN for each slice, not just aggregate delta;
- top high-confidence wrong examples in the best and worst motif slices.

The operator basis claim is only supported if ray/operator masks improve ray-heavy or relation-heavy motifs such as `pin`, `skewer`, and `discovered_attack`, especially at `hard` and `very_hard` difficulty. If gains appear only on `very_easy`, `hanging`, or broad material/eval buckets, treat the idea as mostly a shortcut learner.

## Decision Rule

Proceed only if the model beats the size-matched CNN on PR AUC and improves the near-puzzle row without worsening the declared hard/ray-heavy slices.
