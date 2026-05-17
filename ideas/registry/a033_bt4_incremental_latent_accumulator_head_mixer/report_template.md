# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Accumulator diagnostics: per-block mean and max of the routed-
    token output norm `||Y||`, per-block norm of `h_global` and
    `h_king` (the two board-level latents), and the per-block
    ratio of broadcast-latent contribution to the local-token
    contribution inside the `phi` MLP. Flag mixer collapse (all
    `||Y|| ~ 0`) or stream degeneracy (`h_king ~ 0` or
    `h_global ~ 0`) even when the model is competitive.
  - Soft-anchor diagnostics: per-block entropy of the saliency
    softmax `anchor_w = softmax(Conv2d(C -> 1)(x))` (high entropy
    = uniform anchor over all 64 squares = no real king
    selection; low entropy = the anchor commits to a small subset
    of squares), and per-block correlation between the
    `argmax(anchor_w)` square and the rule-exact own-king square
    from `simple_18` on the held-out test set. If the anchor is
    near-uniform or anti-correlated with the rule-exact king
    square, the soft-anchor mechanism is decorative and the A6
    `zero_king_accumulator` ablation should match this idea.
  - Per-square bias diagnostics: variance of the learned
    `global_square in R^{64 x latent_dim}` rows across squares,
    and the per-square `||global_square_s||` distribution. Flag
    near-zero variance (the per-square bias is decorative; the
    A8 `shuffle_square_order` ablation should match this idea)
    or extreme outliers (one or two squares dominate the global
    stream).
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control -- the same global-pooling
    role with no permutation-structured pooled-accumulate-then-
    broadcast prior; the most informative head-to-head for this
    idea).
  - `p028_incremental_latent_accumulator_head` (A3 head-form
    control with the rule-exact `(12, 64)` piece-plane indicator
    and the rule-exact own-king square that the mixer cannot
    read).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. The rule-exact own-king
    square `k in {0, ..., 64}` is replaced by a learned soft-argmax
    saliency square computed from a `Conv2d(C -> 1)` over the
    channel features. If the learned saliency cannot recover the
    king position from the channel features alone, the mixer
    anchors on the wrong square and the `king_anchor_table` row
    is mis-selected; report the correlation between
    `argmax(anchor_w)` and the rule-exact own-king square from
    `simple_18` alongside the headline number.
  - The per-(piece-type, square) embedding table
    `G in R^{12 x 64 x global_dim}` is replaced by a
    `Linear(C -> latent_dim)` projection of the per-square channel
    features plus a `(64, latent_dim)` per-square bias. The mixer
    cannot distinguish piece types directly; it can only do so
    indirectly via whatever piece-type structure the trunk has
    already encoded into the channel features. If A3 (the
    primitive used as a pooled head with the rule-exact piece-
    type indicator) strictly beats this mixer, the per-(piece-
    type, square) embedding is what made the primitive work,
    not the pooled-accumulate-then-broadcast structure.
  - The `king_anchor_table` is `(64, latent_dim)` instead of the
    source primitive's `(65, 12, 64, king_dim)` HalfKA-style
    indexing. There is no separate "no king" row and no
    sub-indexing by piece type at the anchor square.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream.
  - The board-wide sum `h_global = sum_s g_s` loses scale on very
    late-endgame positions (few non-empty channel features) and
    may saturate on early-game positions (densely packed
    channels). This should show up as flat performance on the
    `crtk_phase: endgame` slice if the king-anchor stream cannot
    compensate.

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
  king-safety and king-zone activity slices where the
  king-anchored accumulator carries decisive context (king-side
  attack motifs, mating-net tactics, exposed-king tactics);
  endgame king-position slices (`crtk_phase: endgame`, especially
  king-and-pawn motifs) where the king-anchored re-pool captures
  geometry the conv windows miss; global-material slices
  (material-imbalance tactics, sacrifice-for-attack motifs) where
  the permutation-invariant pooled sum captures global piece
  count cleanly; and the mid-to-upper `crtk_difficulty` band
  where the trunk's local-receptive-field stack and the dense
  `attention` baseline are both likely to be insufficient.
- Slices where this idea is expected to fail or merely match:
  local-tactical slices (one- or two-square forks, simple captures)
  where the conv mixer's local 3x3 window already saturates;
  opening-phase positions with full piece complement where the
  global pooled sum has very high magnitude and so the
  permutation-invariant accumulator loses discriminative power;
  and the lowest `crtk_difficulty` band where the trunk's exchange
  features already saturate. These should be measured for non-
  regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer A6 `zero_king_accumulator`
  ablation. If either matches this idea on the target slice, the
  king-anchored permutation-structured accumulator prior is not
  load-bearing inside the BT4 tower. A2 (`bt4_attention_mixer` --
  dense all-pairs mixing without the pooled-accumulate-then-
  broadcast prior) is the canonical falsifier: if dense attention
  matches, the permutation-structured pooled accumulator is just
  a cheaper global-pool variant of attention. A7 (`linear_only`)
  closing the slice also kills the non-linear-lift claim.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
