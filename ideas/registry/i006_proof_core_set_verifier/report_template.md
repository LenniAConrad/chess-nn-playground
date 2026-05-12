# Idea Report Template

## Required Comparisons

- random witness selection
- full-board verifier
- no relation features
- bounded versus unbounded residual
- witness budget sweep
- BT4 and size-matched CNN baselines

## Required Metrics

- test F1
- test PR AUC
- 3x2 source-class confusion matrix plus `slice_report_val.md` and `slice_report_test.md`
- near-puzzle false-positive rate
- puzzle recall
- deletion gap by source class
- witness type distribution

## Required Slice Analysis

Follow `ideas/docs/BENCHMARK_REPORTING.md`. In this idea's report, add:

- deletion gap, witness count, and accuracy by `crtk_difficulty`;
- witness type distribution by `crtk_phase`;
- motif slices for `mate_in_1`, `promotion`, `underpromotion`, `pin`, `fork`, `overload`, `hanging`, and `(none)`;
- learned witness selection versus random witness selection deltas per slice;
- high-confidence wrong examples with the selected proof core rendered or listed.

The proof-core claim is supported only if learned sparse witnesses improve hard proof-like motifs and deletion gaps remain large in those same slices. If random witnesses match the learned selector by difficulty and motif, or deletion gaps are large only for easy/hanging positions, reject the proof-core interpretation.

## Decision Rule

Keep the idea only if learned sparse witnesses beat random witnesses and the deletion diagnostic shows the selected proof core is causally used.
