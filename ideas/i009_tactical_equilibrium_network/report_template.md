# Idea Report Template

## Required Comparisons

- BT4 baseline
- size-matched board trunk
- max-attacker-only model
- mean payoff pooling
- legal defenders vs random defenders
- solver step sweep

## Required Metrics

- test F1
- test PR AUC
- test accuracy
- 3x2 source-class confusion matrix plus `slice_report_val.md` and `slice_report_test.md`
- near-puzzle false-positive rate
- puzzle recall
- equilibrium value by source class
- attacker/defender entropy by source class
- exploitability by source class

## Required Slice Analysis

Follow `ideas/BENCHMARK_REPORTING.md`. In this idea's report, add:

- equilibrium value, entropy, exploitability, and accuracy by `crtk_difficulty`;
- the same diagnostics by `crtk_phase`;
- motif slices for `overload`, `pin`, `skewer`, `discovered_attack`, `fork`, `hanging`, `mate_in_1`, and `(none)`;
- legal defenders versus random defenders deltas for each important slice;
- examples where equilibrium modeling vetoed static max-attacker false positives.

The equilibrium claim is supported only if attacker/defender interaction improves contested tactical motifs such as `overload`, `pin`, `skewer`, and `discovered_attack`, especially on `hard`/`very_hard` rows. If max-attacker-only or random defenders match the full solver on those slices, the equilibrium layer is not earning its complexity.

## Decision Rule

This is the practical high-upside idea I would try before full proof-number search. Keep it if equilibrium modeling reduces hard-negative false positives more than static and max-threat baselines.
