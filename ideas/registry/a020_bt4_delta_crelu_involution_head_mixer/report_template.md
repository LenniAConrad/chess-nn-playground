# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks 0..N).
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer`.
  - Saturation-rate diagnostic: fraction of accumulator pre-activations
    landing in each ClippedReLU regime (below 0 / in-range / above
    `clip_max`) per block, summarised over the validation split.
  - Reynolds split contribution: norm of `sym` vs `asym` per-block,
    averaged over the validation split.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p015_delta_crelu_involution_head` (A3 head-form control).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
  - `involution_weight=0` ablation of this idea (A5 control).
  - `clip_max=+inf` ablation of this idea (A6 control).
- Known blockers:
  - The channel-reversal involution is a structural stand-in for the
    piece-plane colour swap. On BT4 channels with no piece semantics,
    the Reynolds split may add only spurious mixing; the A5 ablation
    must be inspected before claiming Reynolds equivariance is
    load-bearing.
  - The simple_18 board lacks halfmove/fullmove counters; this is
    irrelevant for the puzzle_binary classifier but may bias diagnostics
    if the saturation tape correlates with phase.
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
  `crtk_difficulty` upper tail and `crtk_eval_bucket = equal` (positions
  where small per-square deltas matter and the saturation tape carries
  signal), plus colour-symmetric `crtk_tactic_motifs` that benefit
  structurally from the Reynolds split.
- Slices where this idea is expected to fail or merely match:
  `mate_in_1`, opening-`crtk_phase` slices, and asymmetric tactical
  motifs — these are not DCIH's territory and should be measured for
  non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace DCIH
  mixer with `conv`) is the primary erase. A5 (`involution_weight=0`)
  should erase any gain attributable to Reynolds equivariance; A6
  (`clip_max=+inf`) should erase any gain attributable to the
  saturation tape. If neither A5 nor A6 erases the lift, DCIH-specific
  attribution has failed.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by `bt4_attention_mixer`.
