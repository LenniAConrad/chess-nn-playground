# Idea Report Template

## Required Comparisons

- puzzle-only trunk
- legal-only auxiliary
- next-latent-only auxiliary
- full dynamics auxiliary
- random move descriptors
- with and without dynamics summary in final head

## Required Metrics

- test F1
- test PR AUC
- 3x2 source-class confusion matrix plus `slice_report_val.md` and `slice_report_test.md`
- near-puzzle false-positive rate
- puzzle recall
- legal auxiliary accuracy
- next-latent MSE
- runtime per epoch

## Required Slice Analysis

Follow `ideas/all_ideas/docs/BENCHMARK_REPORTING.md`. In this idea's report, add:

- puzzle accuracy, legal auxiliary accuracy, and next-latent MSE by `crtk_difficulty`;
- all three diagnostics by `crtk_phase`, since dynamics errors may differ between openings, middlegames, and endings;
- motif slices for `promotion`, `underpromotion`, `mate_in_1`, `fork`, `pin`, `hanging`, and `(none)`;
- random move descriptor deltas by slice;
- examples where dynamics auxiliary improves a hard prediction and examples where it harms calibration.

The rule-consistent dynamics claim is supported only if legal/dynamics auxiliaries improve hard slices or transfer across phases, not just aggregate loss. If auxiliary accuracy is high but puzzle performance does not improve by difficulty, phase, or motif, the representation is probably learning legal trivia rather than useful puzzle structure.

## Decision Rule

Continue if rule-consistent dynamics improves the near-puzzle row or gives a reusable representation that improves multiple chess classification tasks.
