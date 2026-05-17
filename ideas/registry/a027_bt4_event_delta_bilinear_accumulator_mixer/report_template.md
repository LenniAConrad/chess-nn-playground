# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Bilinear-accumulator diagnostics: per-block mean and max of the
    learned soft occupancy `O_s = sigmoid(w . x_s + b)` on positions
    with many active pieces vs sparse positions; per-block norm of
    the first-order sums `||A||`, `||B||` and the FM-identity pair
    term `||Q||`; ratio `||Q|| / (||A|| ||B||)` (collapses toward
    zero when `U` and `V` align). Saturation (every square reading
    `O = 1` or every square reading `O = 0`) or pair-term collapse
    (`||Q|| ~ 0`) should be flagged even when the model is
    competitive.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p022_event_delta_bilinear_accumulator` (A3 head-form control
    with the piece-plane occupancy that the mixer cannot read).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer's soft-occupancy indicator is learned from generic
    channels rather than read off piece planes. If the indicator
    fails to recover occupancy, the FM-identity pair-term sum is
    contaminated by empty-square contributions and the operator
    degenerates into a noisy broadcast; report the in-mixer
    `zero_occupancy`-style ablation alongside the headline number.
  - The pair term `Q = A (.) B - P` can collapse to zero when the
    `U` and `V` projections learn aligned subspaces (`P ~ A (.) B`);
    inspect block-level `||Q||` statistics before declaring null.
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
  multi-piece-interaction-dependent tactics where second-order
  pair-term coordination is load-bearing -- `crtk_tactic_motifs`
  involving `fork`, `discoveredAttack`, `pin`, `skewer`, `xRayAttack`,
  and `battery`/`doubleAttack` patterns -- and the `crtk_difficulty`
  upper tail where multi-piece coordination dominates over local
  conv windows.
- Slices where this idea is expected to fail or merely match:
  `mate_in_1`, quiet-positional `crtk_tactic_motifs`, and
  opening-`crtk_phase` slices -- these are not where second-order
  pair-term aggregation should dominate and should be measured for
  non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the source-primitive `first_order_only`
  ablation (drop the pair term `Q`). If either matches this idea on
  the target slice, the FM-identity pair term is not load-bearing
  inside the BT4 tower.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
