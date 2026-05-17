# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Incremental-delta-linear diagnostics: per-block norm of the
    global accumulator `||S||` after `LayerNorm`, per-block mean
    and max of the per-square contributions `||W_s x_s||`, and the
    ratio of broadcast-back contribution to own-token contribution
    in the fusion MLP. Flag accumulator collapse (`||S|| ~ 0`) or
    own-token domination (broadcast share `~ 0`) even when the
    model is competitive.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p025_incremental_delta_linear_head` (A3 head-form control with
    the per-(piece-type, square) embedding table that the mixer
    cannot read).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `12 x 64` piece-plane indicator. If the per-square
    linear map fails to recover the per-(piece-type, square)
    statistic, the global accumulator is decorative; report the
    `shuffle_squares`-style ablation alongside the headline number.
  - The global sum `S = sum_s W_s x_s + b_s` can collapse to a
    small norm if the per-square weights cancel, in which case the
    broadcast-back contribution to the fusion MLP is dominated by
    `LayerNorm`'s affine scale and the operator degenerates into a
    per-square own-token MLP; inspect block-level `||S||` before
    declaring null.
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
  slices where a stable per-(piece-type, square) count-style
  statistic discriminates positives from negatives -- simple
  material-count puzzles, back-rank slices, rook-square slices, and
  the lower-to-mid `crtk_difficulty` band where the NNUE-style
  linear-additive accumulator is most informative. Also expected to
  win where the `crtk_phase` is `endgame` because the dense per-
  square count signal aligns with reduced-material reasoning.
- Slices where this idea is expected to fail or merely match:
  multi-piece tactical interactions (fork / pin / skewer / x-ray /
  battery / discovered-attack), deep `crtk_difficulty` tail
  positions, and any slice where a linear additive accumulator is
  obviously insufficient. These should be measured for non-
  regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the source-primitive `zero_accumulator`
  ablation (hold `S = 0`). If either matches this idea on the target
  slice, the linear-additive accumulator is not load-bearing inside
  the BT4 tower.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
