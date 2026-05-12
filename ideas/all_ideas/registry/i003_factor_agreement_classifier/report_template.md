# Idea Report Template

## Required Comparisons

- plain concatenation fusion
- no disagreement penalty
- each single-factor branch
- BT4 and simple CNN benchmark baselines

## Required Metrics

- test F1
- test PR AUC
- 3x2 source-class confusion matrix plus `slice_report_val.md` and `slice_report_test.md`
- near-puzzle false-positive rate
- puzzle recall
- mean disagreement by source class

## Required Slice Analysis

Follow `ideas/all_ideas/docs/BENCHMARK_REPORTING.md`. In this idea's report, add:

- branch agreement and disagreement by `crtk_difficulty`;
- branch agreement and disagreement by `crtk_phase`;
- motif-level rows for `fork`, `pin`, `skewer`, `hanging`, `overload`, `mate_in_1`, `promotion`, and `(none)`;
- separate slice tables for false positives on fine label `1` and false negatives on fine label `2`;
- high-confidence wrong examples where one branch disagreed strongly with the final prediction.

The factor-agreement claim is supported only if disagreement is diagnostic: hard or mixed-motif slices should show meaningful branch conflict before correction, and the agreement penalty should reduce mistakes on those same slices. If all branches agree equally on easy and hard positions, or disagreement is only correlated with confidence collapse, the idea is not teaching useful structure.

## Decision Rule

Continue only if agreement improves hard-negative behavior or gives clear diagnostic information that plain fusion lacks.
