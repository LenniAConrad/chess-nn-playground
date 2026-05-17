# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks 0..N).
  - Per-type attention entropy: histogram of `alpha in Delta^3` over
    the four promotion types {Q, R, B, N}, aggregated per block, to
    detect collapse to a single type or to a near-uniform distribution.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `i246_promotion_aware_head` (A3 head-form control with literal
    pawn-substitution semantics).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer adaptation drops the literal pawn-substitution semantics
    of the source primitive. If the four learned type transforms do not
    diverge in feature space (per-type attention entropy near `log 4`
    everywhere), the fanout is a fancy no-op and the idea cannot beat
    `attention`.
  - Promotion / underpromotion is only ~6% of positions, so an
    aggregate PR AUC delta is the wrong headline metric for this idea —
    judge it on the `crtk_tactic_motifs = promotion` slice.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the cross-attention pooled value has small
    magnitude; inspect block-level activation statistics before
    declaring null.

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
  `crtk_tactic_motifs = promotion` and its `underpromotion` sub-entry
  (the largest per-slice PR-AUC gap in the benchmark and the slice the
  source PAH primitive was designed for).
- Slices where this idea is expected to fail or merely match:
  `mate_in_1`, opening-`crtk_phase` slices, `crtk_eval_bucket = equal`
  positions without near-promotion pawns — these are not PAH's
  territory and should be measured for non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace PAH
  mixer with `conv`). If A1 matches this idea on the promotion slice,
  the mixer is not load-bearing inside the BT4 tower.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by `bt4_attention_mixer`.
