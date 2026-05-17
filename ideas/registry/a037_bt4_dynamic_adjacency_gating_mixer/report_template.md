# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Dynamic-adjacency-gating diagnostics: per-block mean and max of
    the routed-token output norm `||Y||`; per-block per-type gate
    statistics (mean, max, entropy of `g_t(Z_i)` across the 64
    squares and across the batch, for each `t in {KNIGHT, KING,
    RANK, FILE, DIAG, ANTIDIAG}`); per-block per-type contribution
    norm `||g_t * M_t (W_t Z)||` for each `t` (to identify which
    type slot carries the signal); per-block per-type pairwise
    correlation between `(W_t Z)` and `(W_t' Z)` for `t != t'` (to
    detect kernel collapse, where multiple `W_t` learn the same
    map). Flag mixer collapse (all `||Y|| ~ 0`) or single-type
    dominance (one `t` carries `>= 90%` of `||Y||` -- if so, the
    other type slots are decorative and A5 `single_move_type`
    should match this idea).
  - Gate-correlation diagnostics: per-block correlation between
    per-square per-type gates `g_t(Z_i)` and the rule-exact piece
    occupancy maps from `simple_18` (e.g., correlate `g_RANK(Z_i)`
    with the union of own-color rook + queen occupancy, or
    `g_DIAG(Z_i)` with own-color bishop + queen occupancy). If the
    mixer is learning the chess-piece geometry from the channel
    features, `g_t(Z_i)` should correlate non-trivially with the
    own-color piece occupancy of the matching piece type. If it
    does not, the per-square per-type content gate is decorative
    and A6 `uniform_gate` should match this idea.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control -- the same all-pairs
    long-range mixing role with no chess-rule per-type prior; the
    most informative head-to-head for this idea).
  - `p032_dynamic_adjacency_gating` (A3 head-form control with the
    occupancy-blocked position-specific binary adjacency, the
    pawn_push and pawn_capture move-type slots, the pooled
    feature-vector read-out, and the trunk-fusion MLPs that the
    mixer cannot use).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. The per-square per-type gate
    has no direct access to "which square holds a sliding piece of
    type `t`"; it must rediscover that from whatever piece-occupancy
    structure the trunk has already encoded into the channel
    features. If the trunk under-encodes piece occupancy in early
    blocks, the per-square per-type gate cannot learn content-
    dependent type weighting in those blocks; the operator
    degenerates to a constant-gate per-type aggregation over the
    static chess-rule reach geometry.
  - The static per-type adjacency does not include occupancy-based
    blocker resolution. Rooks "see" through their own piece on the
    same file; bishops "see" through their own piece on the same
    diagonal. If A3 (the primitive used as an additive head with the
    occupancy-blocked position-specific adjacency) strictly beats
    this mixer, the absence of occupancy blocking is the load-
    bearing cost of the adaptation.
  - The pawn_push and pawn_capture move-type slots from the source
    primitive are dropped (`T = 6` here vs `T = 8` in the source
    primitive's head form) because the BT4 mixer cannot distinguish
    side-to-move from the channel features alone. If pawn-tactic
    slices (passed-pawn, pawn-sac, pawn-fork motifs) under-perform
    the source primitive, the dropped slots are load-bearing.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The `T = 6` per-type dense matmuls are the dominant per-block
    cost (`O(6 * 64^2 * C)` per sample). They are parallel across
    types (one batched-einsum implementation across all types
    simultaneously), so the wall-clock cost on a GPU with enough
    memory is closer to one large batched-einsum than to `T`
    sequential matmuls. If throughput on matched hardware falls
    below ~50% of the `conv` baseline, the matched comparison must
    drop `model.num_blocks` or `model.channels` to compensate (see
    `trainer_notes.md`).

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
  single-move-type-dominated tactical slices where one type of
  chess-rule reach drives the tactic (open-file / rook-on-open-file
  motifs for the `FILE` slot; long-diagonal / bishop-pair-on-long-
  diagonal / fianchetto motifs for `DIAG` and `ANTIDIAG`; knight-
  outpost / knight-fork motifs for `KNIGHT`; king-safety / opposing-
  king / king-and-pawn-endgame motifs for `KING`; rank-pin /
  back-rank-mate motifs for `RANK`); and the mid `crtk_difficulty`
  band where the trunk's local-receptive-field stack saturates but
  before dense attention's all-pairs mixing dominates.
- Slices where this idea is expected to fail or merely match:
  pawn-tactic slices (passed-pawn, pawn-sac, pawn-fork) where the
  dropped `PAWN_PUSH` and `PAWN_CAPTURE` types of the source
  primitive carry the signal; local-tactical slices (one- or two-
  square forks, simple captures) where the conv mixer's local 3x3
  window already saturates; positions where multiple move types
  interact tightly (pin-on-pin, X-ray attacks) where the per-type
  decomposition fights the multi-hop structure that
  `legal_move_laplacian_resolvent` (a036) targets; the lowest
  `crtk_difficulty` band where the trunk's exchange features
  already saturate. These should be measured for non-regression,
  not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer A5 `single_move_type` and A6
  `uniform_gate` ablations. If any of these matches this idea on
  the target slice, the per-move-type masked aggregation prior is
  not load-bearing inside the BT4 tower. A2 (`bt4_attention_mixer`
  -- dense all-pairs mixing without the chess-rule per-type prior)
  is the canonical falsifier: if dense attention matches, the
  chess-rule per-type decomposition is just a constrained form of
  attention. A8 (`shuffle_adjacency`) closing the slice also kills
  the chess-rule reach prior claim.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
