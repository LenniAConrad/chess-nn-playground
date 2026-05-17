# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks 0..N).
    The DPA mixer's alignment-restricted pair message is expected to
    produce structured rather than full-rank activations; report it
    explicitly so a null result can be interpreted as "the alignment
    geometry is redundant given the stem conv's receptive field"
    rather than "the mixer broke training".
  - Pair-message energy by edge type (rook, bishop, file, rank,
    diagonal) summed across blocks: a non-trivial DPA mixer should
    show non-uniform energy concentrated on the chess-meaningful edge
    types for the target slice.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p014_delta_pair_accumulator` (A3 head-form control).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
  - DPA-mixer with all-pairs mask (A5 debug-time control).
- Known blockers:
  - The static-fixed-point DPA adapter omits the make/unmake
    autograd path that defines the primitive at inference time. A
    null here is not a falsifier for the delta-stream variant of DPA;
    say so in the conclusion.
  - The piece-type pair table ``W_{type(i),type(j)}`` from the source
    primitive is replaced by a content-bilinear over `C` channels in
    this adapter. If the slice-level lift depends on piece-type
    indexing, this adapter cannot recover it; report whether the gap
    closes when the head-form (A3) is used.
  - The alignment mask is position-independent here because all 64
    squares are always "active"; the source's occupancy-conditioned
    subset is not recoverable from arbitrary `C` channels. Mention
    this caveat in the conclusion.
  - The simple_18 board lacks halfmove/fullmove counters; this is
    irrelevant for the puzzle_binary classifier but may bias
    diagnostics if the pair message's saturated state depends on
    phase metadata.

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
  `crtk_tactic_motifs` slices that depend on rook/bishop-style line
  geometry (open files, long diagonals, batteries) and middlegame
  `crtk_phase` slices where rank/file/diagonal alignments dominate.
- Slices where this idea is expected to fail or merely match:
  `mate_in_1`, knight-fork-dominated `crtk_tactic_motifs`, and
  opening-phase slices — these depend on non-aligned local
  interactions that an alignment-restricted pair mixer is unlikely
  to recover; measure for non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace DPA
  mixer with `conv`). If A1 matches this idea on the target slice, the
  mixer is not load-bearing inside the BT4 tower. A5 (alignment mask
  replaced by all-pairs mask) should partially close the target-slice
  gap if the chess geometry is doing real work.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by `bt4_attention_mixer`.
