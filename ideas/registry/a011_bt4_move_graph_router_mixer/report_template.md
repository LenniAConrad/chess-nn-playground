# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    0..N).
  - MGR adjacency diagnostics on a held-out batch: average
    per-source degree (should sit near `64 * edge_density`),
    fraction of empty source rows (should be near 0 with the
    `clamp_min(1.0)` floor), and Jaccard overlap between the
    content-derived adjacency at block 0 and the rule-derived
    legal-move adjacency derived from the input simple_18 (sanity
    check that the content-derived mask is not pure noise).
  - Edge-MLP diagnostics: per-block norm of `phi_theta([x_i, x_j])`
    before vs after masking, to detect "mask is irrelevant" failure
    where the unmasked edge message already carries the full
    contribution.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` (the MGR mixer is expected to be
    several times slower per step than conv at default capacity).
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p006_move_graph_router` (A3 head-form control).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
  - Random-mask MGR (A6 control).
- Known blockers:
  - The MGR mixer is more expensive per call than a 3x3 conv. If
    throughput falls below ~40% of the conv baseline at scout scale,
    matched-budget claims become unsound; drop tower capacity or
    rewrite as additive head.
  - The mixer substitutes a content-derived adjacency for the source
    primitive's rule-derived legal-move adjacency; if the Jaccard
    overlap with the rule-derived adjacency is near zero across all
    blocks, the "move-graph router" name is only a rebrand for a
    sparse content-derived mixer at this layer.
  - The simple_18 board lacks halfmove/fullmove counters; this is
    irrelevant for the puzzle_binary classifier but may bias
    diagnostics if the legal-move geometry depends on move-count
    context.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the mixer's output magnitude is small; inspect
    block-level activation statistics before declaring null.

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an
aggregate confusion matrix. Every promoted idea must require:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives
  for fine label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and
  motifs;
- a short conclusion describing what the model appears able and
  unable to learn.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  high-`crtk_difficulty` puzzles whose resolution rests on ray-style
  information propagation (pins, x-rays, battery threats, skewers)
  that a single 3x3 conv cannot reach, and `crtk_tactic_motifs`
  dominated by long-range sliding-piece tactics where conv and
  attention should under-localise. Also `crtk_phase = middlegame`
  positions with dense legal-move graphs.
- Slices where this idea is expected to fail or merely match:
  `mate_in_1` and opening-`crtk_phase` slices where the decisive
  move is obvious from a single-square local pattern -- these should
  be measured for non-regression, not for lift. Closed positions
  with very few legal moves where the per-source aggregator averages
  a handful of edges and the mixer reduces to noise.
- Ablation that should erase the slice-level gain: A1 (replace MGR
  mixer with `conv`). If A1 matches this idea on the target slice,
  the mixer is not load-bearing inside the BT4 tower. Secondary:
  A6 (random-mask MGR) -- if a uniform Bernoulli mask matches the
  learned-mask MGR, the content-derived adjacency carries no signal.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
