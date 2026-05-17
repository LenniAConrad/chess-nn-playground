# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Move-kernel-operator diagnostics: per-block mean and max of the
    routed-token output norm `||Y||`; per-block per-type
    contribution norm `||M_t (W_t Z)||` for each
    `t in {KNIGHT, RANK, FILE, DIAG, ANTIDIAG, KING}` (to identify
    which type slot carries the signal); per-block per-type
    pairwise correlation between `(W_t Z)` and `(W_t' Z)` for
    `t != t'` (to detect kernel collapse, where multiple `W_t`
    learn the same map). Flag mixer collapse (all `||Y|| ~ 0`) or
    single-type dominance (one `t` carries `>= 90%` of `||Y||` --
    if so, the other type slots are decorative and A5
    `shared_kernel` should match this idea).
  - Per-block per-type singular-value spectrum of `W_t` (to detect
    rank collapse of the per-type projection and to flag whether a
    scalar `w_t * I` could replace `W_t`; if the leading singular
    value dominates by `>= 10x` for every `t`, A6 `scalar_per_type`
    should match this idea).
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control -- the same all-pairs
    long-range mixing role with no chess-rule per-type prior; the
    most informative head-to-head for this idea).
  - `p033_move_kernel_operator` (A3 head-form control with the
    original `Linear(13)` per-square seed feature, the pooled
    feature-vector read-out, and the trunk-fusion MLPs that the
    mixer cannot use).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. The per-square input is
    therefore the trunk's learned channel feature, not the source
    primitive's `Linear(13)` projection of piece planes. If A3 (the
    primitive used as an additive head with the original per-square
    seed) strictly beats this mixer, the per-square seed
    substitution is the load-bearing cost of the adaptation.
  - The static per-type adjacency does not include occupancy-based
    blocker resolution (by design -- this matches the source
    primitive's framing). Rooks "see" through their own piece on
    the same file; bishops "see" through their own piece on the
    same diagonal. The mixer therefore mixes signal from squares
    behind blockers indiscriminately; the trunk must learn to
    down-weight them through the per-type matrix `W_t` or the
    surrounding SqueezeExcite.
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
  pawn-tactic slices (passed-pawn, pawn-sac, pawn-fork) where MKO
  has no pawn-move type (the source primitive's six-type schema
  does not separate pawn moves from king/sliding types); local-
  tactical slices (one- or two-square forks, simple captures) where
  the conv mixer's local 3x3 window already saturates; positions
  with sliding pieces behind own-color blockers (X-ray attacks,
  back-rank covered by a friendly piece) where MKO's occlusion-free
  reach over-connects the rays; the lowest `crtk_difficulty` band
  where the trunk's exchange features already saturate. These
  should be measured for non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer A5 `shared_kernel` and A6
  `scalar_per_type` ablations. If any of these matches this idea on
  the target slice, the per-move-type matrix specialisation prior
  is not load-bearing inside the BT4 tower. A2
  (`bt4_attention_mixer` -- dense all-pairs mixing without the
  chess-rule per-type prior) is the canonical falsifier: if dense
  attention matches, the chess-rule per-type decomposition is just
  a constrained form of attention. A7 (`shuffle_features`) closing
  the slice kills the per-square seed signal claim; A8
  (`uniform_adjacency`) closing the slice kills the chess-rule
  reach prior claim.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
