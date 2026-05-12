# Ablations

- A1: `solver_cycles: 1` tests whether one projection sweep is enough.
- A2: use `training.loss: bce_with_logits` with the same model to test whether residual shaping matters.
- A3: hide trace diagnostics from the readout to test whether the solver trace is the signal.
- A4: remove endpoint/closure projections to test whether role-relation consistency matters.
- A5: disable binary hard-negative weighting.

Falsify or demote the idea if a parameter-matched LC0 BT4 classifier matches PR AUC and matched-recall near-puzzle false positives, or if projection diagnostics do not separate true puzzles from near-puzzle negatives.
