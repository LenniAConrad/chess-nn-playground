# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks 0..N).
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p004_tail_copula_concordance` (A3 head-form control).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The TCC mixer is more expensive per call than a 3x3 conv because
    it runs an `O(N^2 * C)` pairwise soft-rank pass per board. If
    throughput falls below ~40% of the conv baseline at scout scale,
    matched-budget claims become unsound; drop tower capacity or
    rewrite as additive head.
  - The simple_18 board lacks halfmove/fullmove counters; this is
    irrelevant for the puzzle_binary classifier but may bias diagnostics
    if the mixer's evidence-channel projection depends on phase metadata.
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
  `crtk_difficulty` upper tail (positions whose resolution depends on
  cross-site evidence alignment that separates real tactics from
  marginal-matched near-puzzles) and `crtk_eval_bucket = equal`
  (positions where the static value sum is uninformative and the
  tail-concordance structure of evidence channels is load-bearing).
- Slices where this idea is expected to fail or merely match:
  `mate_in_1`, promotion-heavy `crtk_tactic_motifs`, opening-`crtk_phase`
  slices — these are not TCC's territory and should be measured for
  non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace TCC
  mixer with `conv`). If A1 matches this idea on the target slice, the
  mixer is not load-bearing inside the BT4 tower.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by `bt4_attention_mixer`.
