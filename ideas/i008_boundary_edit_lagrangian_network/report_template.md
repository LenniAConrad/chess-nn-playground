# Idea Report Template

## Required Comparisons

- BT4 baseline
- base-only encoder
- null-move-only edit basis
- random edit basis
- direct energy head without solver
- `E_plus` only
- `E_minus` only

## Required Metrics

- test F1
- test PR AUC
- test accuracy
- 3x2 source-class confusion matrix plus `slice_report_val.md` and `slice_report_test.md`
- near-puzzle false-positive rate
- puzzle recall
- `E_plus` by source class
- `E_minus` by source class
- top edit family distribution

## Required Slice Analysis

Follow `ideas/BENCHMARK_REPORTING.md`. In this idea's report, add:

- `E_plus`, `E_minus`, accuracy, and false-positive rate by `crtk_difficulty`;
- edit energy by `crtk_eval_bucket`, because boundary energy can collapse into eval-margin prediction;
- `crtk_phase` slices for opening, middlegame, and endgame boundary behavior;
- motif slices for `promotion`, `underpromotion`, `mate_in_1`, `pin`, `overload`, `hanging`, and `(none)`;
- top edit families inside the worst false-positive and false-negative slices.

The boundary-edit claim is supported only if edit energies explain boundary cases: `hard`/`very_hard`, equal/slight buckets, near-puzzle false positives, and promotion/endgame motifs. If `E_plus`/`E_minus` mostly reproduce `crtk_eval_bucket` or material imbalance and do not improve these slices, reject the boundary-energy mechanism.

## Decision Rule

This idea is worth implementing if we want a model that explicitly understands near-puzzles as boundary cases. Keep it only if legal edit energy gives a measurable hard-negative advantage over null-move-only and base-only models.
