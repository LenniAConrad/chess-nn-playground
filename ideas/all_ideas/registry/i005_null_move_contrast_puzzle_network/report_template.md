# Idea Report Template

## Required Comparisons

- current-only shared encoder
- current/null concat without explicit delta
- deterministic null view
- random side-swap view
- BT4 and size-matched CNN baselines

## Required Metrics

- test F1
- test PR AUC
- test accuracy
- 3x2 source-class confusion matrix plus `slice_report_val.md` and `slice_report_test.md`
- near-puzzle false-positive rate
- puzzle recall
- mean tempo contrast by source class

## Required Slice Analysis

Follow `ideas/all_ideas/docs/BENCHMARK_REPORTING.md`. In this idea's report, add:

- tempo-contrast mean, accuracy, and false-positive rate by `crtk_difficulty`;
- tempo-contrast mean and calibration by `crtk_phase`;
- motif slices for `mate_in_1`, `fork`, `discovered_attack`, `overload`, `pin`, `hanging`, and `(none)`;
- deterministic null view versus random side-swap view deltas for each important slice;
- high-confidence wrong examples where null contrast was large but the prediction was wrong.

The null-move claim is supported only if deterministic tempo contrast improves tempo-sensitive motifs and harder rows, not merely `very_easy` positions. If random side-swap gives the same gains by difficulty and motif, the model is using generic augmentation noise rather than chess tempo information.

## Decision Rule

Keep the idea only if deterministic tempo contrast reduces near-puzzle mistakes or reveals a strong diagnostic split between puzzle and near-puzzle rows.
