# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Sparse-routing diagnostics: per-block mean and max of the
    routed-token output norm `||Y||`, per-block average attention
    entropy on the on-support edges (high entropy = uniform sparse
    pool; low entropy = the router commits to a small subset of
    legal-move edges), and the per-block share of attention mass
    flowing along each chess-direction class (rook rays, bishop
    rays, knight jumps, king/pawn steps) relative to the residual
    path. Flag mixer collapse (all `||Y|| ~ 0`) or
    direction-class degeneracy (one class dominates) even when the
    model is competitive.
  - Edge-gate diagnostics: per-block distribution of the learned
    `g = sigmoid(theta)` over the on-support entries -- mean,
    fraction of edges with `g > 0.5`, and per-block correlation
    between `g` and the rule-exact legal-move adjacency on the
    held-out test set. If `g` is near-uniform or anti-correlated
    with the rule-exact adjacency, the `full_64x64_mask` ablation
    should match this idea.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control -- the same masked-attention
    backbone with no sparsity prior; the most informative
    head-to-head for this idea).
  - `p027_sparse_legal_move_router_head` (A3 head-form control with
    the rule-exact piece-plane legal-move adjacency that the mixer
    cannot read).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. The static chess-geometry
    support `S` ignores blockers (slider rays are unobstructed by
    construction), so the learned per-edge gate
    `g = sigmoid(theta)` must recover the blocker structure from
    the channel features alone. If it cannot, the mixer routes
    signal through impossible (blocked) moves; report the
    correlation between `g` and the rule-exact legal-move adjacency
    alongside the headline number.
  - The dense `(B, 64, 64)` matmul-then-mask costs the same FLOPs
    as the `bt4_attention_mixer` baseline. The sparsity here is a
    *prior on attention structure*, not a sparse-matmul
    optimisation; wall-clock cost is not lower than dense attention
    and may be slightly higher due to the gate computation and the
    log-bias add.
  - The learned per-edge gate is a `(64, 64) = 4096`-parameter
    table; that is a small additional capacity bump versus
    `bt4_attention_mixer`. The A4 capacity-matched conv comparison
    must account for it.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream.

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
  multi-target tactical slices where information routing along
  legal-move edges is more discriminative than dense all-pairs
  attention -- fork-style motifs (one piece routes to two distinct
  legal targets), discovered attacks, pins, skewers, and
  sliding-piece long-range threats. Also expected to win on
  `crtk_phase` `middlegame` slices where dense piece activity
  rewards a structured sparse mask, and on the mid-to-upper
  `crtk_difficulty` band where the trunk's local-receptive-field
  stack and the dense `attention` baseline are both likely to be
  insufficient.
- Slices where this idea is expected to fail or merely match:
  king-stripped endgames with very few legal moves (the routed
  feature is mostly self-loops and carries no real information),
  opening-phase positions where most legal-move edges are
  blocked by densely-packed pieces (the static support
  over-counts edges that the learned gate must suppress), and
  the lowest `crtk_difficulty` band where the trunk's exchange
  features already saturate. These should be measured for non-
  regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer `full_64x64_mask` ablation
  (drop the chess-geometry support). If either matches this idea
  on the target slice, the chess-structured sparse-routing prior
  is not load-bearing inside the BT4 tower. A2
  (`bt4_attention_mixer` -- the same masked-attention backbone
  without the sparsity prior) is the canonical falsifier: if
  dense attention matches, the sparsity is decorative. A6
  (`shuffle_adjacency`) closing the slice also kills the
  chess-content claim.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
