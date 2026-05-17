# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks 0..N).
  - Blocker-reset diagnostics: per-block mean and max of the learned
    soft occupancy `O_s = sigmoid(w . x_s + b)` on positions with at
    least one slider vs positions without sliders; per-direction
    decay `lambda_d = sigmoid(decay_logit)` mean across hidden units;
    distribution of the per-direction ray summary magnitude
    `||y_{s,d}||`. Saturation (every square reading `O = 1` or every
    square reading `O = 0`) should be flagged even when the model is
    competitive.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p020_blocker_reset_ray_scan` (A3 head-form control with the
    piece-plane occupancy that the mixer cannot read).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer's soft-occupancy indicator is learned from generic
    channels rather than read off piece planes. If the indicator
    fails to recover occupancy, the hard reset gate is uninformative
    and the operator degenerates into a directional linear smear;
    report the in-mixer `zero_blocker` ablation alongside the headline
    number.
  - The depth-`L` Python-loop scan across 8 directions limits CUDA
    parallelism; if throughput falls below ~40% of the conv baseline
    at scout scale, matched-budget claims become unsound. Drop tower
    capacity or rewrite as additive head.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the mixer's output magnitude is small; inspect
    block-level activation statistics before declaring null.

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an aggregate
confusion matrix. Every promoted idea must require:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for
  fine label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and
  motifs;
- a short conclusion describing what the model appears able and unable
  to learn.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  sliding-piece-dependent tactics where ray geometry with a blocker
  reset is load-bearing -- `crtk_tactic_motifs` involving `pin`,
  `skewer`, `discoveredAttack`, `xRayAttack`, rook-on-open-file, and
  queen-line-into-king-zone patterns -- and the `crtk_difficulty`
  upper tail where long-range ray interactions matter more than local
  conv windows.
- Slices where this idea is expected to fail or merely match:
  `mate_in_1`, quiet-positional `crtk_tactic_motifs`, and
  opening-`crtk_phase` slices -- these are not where blocker-reset ray
  scanning should dominate and should be measured for non-regression,
  not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`). If A1 matches this idea on the target slice, the
  mixer is not load-bearing inside the BT4 tower.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by `bt4_attention_mixer`.
