# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    0..N).
  - ARSA attention diagnostics on a held-out batch:
    `arsa_attention_entropy` distribution (high entropy = uniform
    over the 9 slots; low = routing concentrated on one direction),
    `arsa_self_weight` mean (high values would indicate the
    operator is mostly attending to its own square -- failure
    mode), `arsa_blocker_count` per source (how many of the 8 ray
    slots actually have a blocker vs are masked out).
  - First-blocker fidelity probe: on the same held-out batch,
    compare the content-derived first-blocker indices at block 0
    against the rule-derived first-blocker indices computed from
    the input `simple_18` occupancy plane. Report mean Jaccard
    overlap per source square. Near-zero overlap means the
    content-derived occupancy at the BT4 token grid is unrelated to
    the rule-derived occupancy the source primitive was validated
    against.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer` (ARSA
    is expected to sit between the two: cheaper than dense
    `O(B * 64^2 * d)` attention because `K = 9`, more expensive
    per block than a 3x3 conv because of the q/k/v projections and
    the masked softmax).
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p007_attack_ray_sparse_attention` (A3 head-form control).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
  - `random_keys` ARSA (A5 control).
  - `uniform_attention` ARSA (A6 control).
- Known blockers:
  - The ARSA mixer substitutes a content-derived soft occupancy
    for the source primitive's rule-derived occupancy. If the
    first-blocker-fidelity Jaccard overlap with the rule-derived
    indices is near zero across all blocks, the "attack-ray sparse
    attention" name is only a rebrand for a 9-slot sparse content-
    derived mixer at this layer.
  - The 9-slot cardinality is small; if every per-query softmax
    saturates on the direction bias, gradients to `q_proj` /
    `k_proj` collapse and the operator degrades to a
    direction-biased mean pool.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the mixer's output magnitude is small; inspect
    block-level activation statistics before declaring null.
  - The simple_18 board lacks halfmove/fullmove counters; this is
    irrelevant for the puzzle_binary classifier but may bias
    diagnostics if the first-blocker geometry depends on move-count
    context.

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
  high-`crtk_difficulty` puzzles whose resolution rests on first-
  blocker / sliding-piece information (pins, x-rays, skewers,
  discovered attacks, battery threats) that a single 3x3 conv cannot
  reach in one mixer call, and `crtk_tactic_motifs in
  {pin, skewer, x_ray, discovered_attack, battery}`. Also
  `crtk_phase = middlegame` positions with many sliding pieces on
  open files / diagonals.
- Slices where this idea is expected to fail or merely match:
  `mate_in_1` and opening-`crtk_phase` slices where the decisive
  move is obvious from a single-square local pattern -- these should
  be measured for non-regression, not for lift. Closed positions
  with very few sliding pieces where most rays have no blocker and
  the 9-slot softmax collapses to the self-edge.
- Ablation that should erase the slice-level gain: A1 (replace ARSA
  mixer with `conv`). If A1 matches this idea on the target slice,
  the mixer is not load-bearing inside the BT4 tower. Secondary:
  A5 (random-keys ARSA) -- if random per-source 8-square draws
  match the rule-derived first-blocker indices, the chess-specific
  geometry carries no signal.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
