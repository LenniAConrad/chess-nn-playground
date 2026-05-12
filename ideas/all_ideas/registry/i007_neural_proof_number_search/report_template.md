# Idea Report Template

## Required Comparisons

- BT4 baseline
- size-matched board-only trunk
- response-minimax depth-1/2 equivalent
- depth 1 vs depth 2 vs depth 3
- proof-number aggregation vs mean tree pooling
- tactical beam vs random legal beam
- with and without AND/OR roles

## Required Metrics

- test F1
- test PR AUC
- test accuracy
- 3x2 source-class confusion matrix plus `slice_report_val.md` and `slice_report_test.md`
- near-puzzle false-positive rate
- puzzle recall
- runtime per epoch
- root proof/disproof gap by source class

## Required Slice Analysis

Follow `ideas/all_ideas/docs/BENCHMARK_REPORTING.md`. In this idea's report, add:

- proof/disproof gap, accuracy, recall, and runtime by `crtk_difficulty`;
- proof/disproof gap by `crtk_phase`, especially endgame rows;
- motif slices for `mate_in_1`, `promotion`, `underpromotion`, `overload`, `pin`, `skewer`, `fork`, and `(none)`;
- depth-1/2/3 deltas inside each high-value slice;
- examples where deeper proof search changed a wrong shallow prediction into a correct one, and the reverse.

The proof-number claim is supported only if extra bounded search helps `hard`/`very_hard`, equal/slight eval buckets, and proof-like motifs enough to justify runtime. If depth mostly improves easy tactical rows or random legal beams match tactical beams by slice, this is not a useful proof-search path.

## Decision Rule

This is intended to be a benchmark-breaking architecture. Keep it only if it clearly beats BT4 or gives uniquely strong evidence that bounded neural proof search is the right next path.
