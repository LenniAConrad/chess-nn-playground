# Idea Report Template

## Required Comparisons

- board-only trunk
- actions-only without replies
- reply-count-only baseline
- real replies vs random reply tokens
- mean pooling vs soft minimax

## Required Metrics

- test F1
- test PR AUC
- 3x2 source-class confusion matrix plus `slice_report_val.md` and `slice_report_test.md`
- near-puzzle false-positive rate
- puzzle recall
- runtime per epoch

## Required Slice Analysis

Follow `ideas/all_ideas/docs/BENCHMARK_REPORTING.md`. In this idea's report, add:

- accuracy, puzzle recall, and near-puzzle false-positive rate by `crtk_difficulty`;
- `crtk_phase` slices, because reply semantics may matter differently in openings, tactical middlegames, and simplified endings;
- motif slices for `mate_in_1`, `fork`, `overload`, `pin`, `skewer`, `discovered_attack`, and `(none)`;
- slice-level deltas for real replies versus random reply tokens;
- examples where soft minimax changed the prediction relative to mean pooling, grouped by difficulty and motif.

The response-minimax claim is supported only if legal reply structure improves slices where defensive resources matter: `overload`, `pin`, `skewer`, `discovered_attack`, equal/slight eval buckets, and `hard`/`very_hard` rows. If the model mainly improves easy forcing positions or loses to random reply tokens on the same slices, reject the semantic reply claim.

## Decision Rule

Continue only if real reply semantics and soft minimax both matter.
