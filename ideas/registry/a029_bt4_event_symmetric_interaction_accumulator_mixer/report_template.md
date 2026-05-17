# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Elementary-symmetric diagnostics: per-block mean and max of the
    learned soft occupancy `O_s = sigmoid(w . x_s + b)` on positions
    with many active pieces vs sparse positions; per-block norm of
    each symmetric state `||E^{(r)}||` for `r = 1 .. R` (default
    `R = 2`); ratio `||E^{(2)}|| / ||E^{(1)}||^2` (collapses toward
    zero when tokens have aligned phase or when occupancy
    saturates). Saturation (every square reading `O = 1` or every
    square reading `O = 0`) or higher-order collapse (`||E^{(>=2)}||
    ~ 0`) should be flagged even when the model is competitive.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p024_event_symmetric_interaction_accumulator` (A3 head-form
    control with the piece-plane occupancy that the mixer cannot
    read).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
  - Order sweep `order in {1, 2, 3}` (A5 control on the mixer-local
    `order` knob).
- Known blockers:
  - The mixer's soft-occupancy indicator is learned from generic
    channels rather than read off piece planes. If the indicator
    fails to recover occupancy, the higher-order states are
    contaminated by empty-square contributions and the operator
    degenerates into a noisy broadcast; report the in-mixer
    `zero_occupancy`-style ablation alongside the headline number.
  - The higher-order states `E^{(>=2)}` can collapse to zero when
    the per-square tokens have aligned phase under the inner
    LayerNorm, leaving the recurrence dominated by `E^{(1)}`;
    inspect block-level `||E^{(r)}||` statistics before declaring
    null.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the mixer's output magnitude is small.

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an
aggregate confusion matrix. Every promoted idea must require:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`,
  `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives
  for fine label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and
  motifs;
- a short conclusion describing what the model appears able and
  unable to learn.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  multi-piece-interaction-dependent tactics where second- and
  third-order coordination is load-bearing -- `crtk_tactic_motifs`
  involving `fork`, `discoveredAttack`, `doubleAttack`, `pin`,
  `skewer`, `xRayAttack`, and `battery` patterns -- and the
  `crtk_difficulty` upper tail where three-piece coordination
  dominates over local conv windows.
- Slices where this idea is expected to fail or merely match:
  `mate_in_1`, quiet-positional `crtk_tactic_motifs`, and
  opening-`crtk_phase` slices -- these are not where higher-order
  symmetric aggregation should dominate and should be measured for
  non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the order sweep A5 forced to `order = 1`
  (drop the higher-order states). If either matches this idea on
  the target slice, the higher-order multiplicative interactions
  are not load-bearing inside the BT4 tower.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
