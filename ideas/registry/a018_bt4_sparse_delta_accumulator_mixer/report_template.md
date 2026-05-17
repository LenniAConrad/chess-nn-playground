# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks 0..N).
    The accumulator-style mixer is expected to produce low effective
    rank; report it explicitly so a null result can be interpreted as
    "the global-summary signal is already in SqueezeExcite" rather
    than "the mixer broke training".
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p013_sparse_delta_accumulator` (A3 head-form control).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The static-fixed-point SDA adapter omits the make/unmake
    autograd path that defines the primitive at inference time. A
    null here is not a falsifier for the delta-stream variant of SDA;
    say so in the conclusion.
  - The SDA mixer's only cross-square interaction is an all-to-all
    uniform sum followed by a broadcast back. This overlaps in
    function with the SqueezeExcite block in the surrounding BT4
    wrapper; if SE is doing the same job, the mixer will appear
    inert. Inspect block-level activation statistics before
    declaring null.
  - The simple_18 board lacks halfmove/fullmove counters; this is
    irrelevant for the puzzle_binary classifier but may bias
    diagnostics if the mixer's saturated state depends on phase
    metadata.

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
  `crtk_eval_bucket = equal` and `crtk_phase = endgame` slices where a
  global, accumulator-style summary of board occupancy is most likely
  to be load-bearing.
- Slices where this idea is expected to fail or merely match:
  `mate_in_1`, sharp-tactical `crtk_tactic_motifs`, and opening-phase
  slices — these depend on local interaction structure that an all-
  to-all uniform mix is unlikely to recover; measure for non-
  regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace SDA
  mixer with `conv`). If A1 matches this idea on the target slice, the
  mixer is not load-bearing inside the BT4 tower.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by `bt4_attention_mixer`.
