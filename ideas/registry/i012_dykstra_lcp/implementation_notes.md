# Implementation Notes

Reusable code lives in `src/chess_nn_playground/models/dykstra_lcp.py`.

Training support adds `DykstraLCPLoss` in `src/chess_nn_playground/training/losses.py` and a small trainer branch for `training.loss: dykstra_lcp`.

The loss uses only binary labels:

- BCE with optional balanced positive class weight;
- online hard-negative weighting selected from binary negatives by current BCE, not by fine label;
- a positive residual term that encourages verified puzzles to be close to the projected feasible certificate;
- a modest negative margin term that discourages all negatives from projecting into easy certificates;
- a trace decay stabilizer.

Fine labels remain diagnostics only through the shared reporting pipeline.

Projector v2 proofread fixes:

- `_simplex_projection` now returns nonnegative rows that sum to one after the positivity clamp.
- Role-budget projection now uses a learnable motif-to-role budget matrix, so `M` controls the budget constraints instead of being ignored through its invariant mean.
- Compactness now uses motif-conditioned budget components.
- Closure now increases bounded slack for unexplained target-role mass before clipping the target role, making slack a measured solver-failure channel.
- The Dykstra configs include `model.architecture_version: 2` and new run names so `--skip-existing` does not reuse old projector results.
