# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Legal-move-Laplacian resolvent diagnostics: per-block mean and
    max of the routed-token output norm `||Y||`, per-block
    distribution of the per-square content weight
    `w(x) = softplus(MLP(X))` (mean, max, entropy across the 64
    squares and across the batch), per-block effective
    `alpha = alpha_init * tanh(alpha_logit)`, and per-block per-
    Neumann-term contribution norm
    `||alpha^k L^k X||` for `k = 0..K` (to identify which terms in
    the truncated series carry the signal). Flag mixer collapse
    (all `||Y|| ~ 0`) or single-hop dominance (only the `k = 0`
    and `k = 1` terms contribute non-trivially -- if so, the
    `k >= 2` Neumann terms are decorative and A5 `k1_gat_rebrand`
    should match this idea).
  - Spectral safety diagnostics: per-block max row-degree of the
    weighted adjacency `W(x) = diag(w(x)) @ A_static` and per-block
    `lambda_max(L)` estimate (top-K Lanczos on a sampled batch).
    If `|alpha * lambda_max| >= 0.9` even with the row-degree
    rescaling, the Neumann partial sum is on the edge of
    divergence and the conservative `alpha_init = 0.25` may need
    to be reduced.
  - Adjacency-correlation diagnostics: per-block correlation
    between per-square content weights `w(x)` and the rule-exact
    piece occupancy maps from `simple_18` (e.g., correlate `w(x)`
    with the union of own-color sliding pieces, or with the king
    square). If the mixer is learning the chess-piece geometry
    from the channel features, `w(x)` should correlate non-
    trivially with the own-color piece occupancy. If it does not,
    the per-square content prior is decorative and A7
    `uniform_piece_weights` should match this idea.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control -- the same all-pairs
    long-range mixing role with no chess-rule legal-move prior; the
    most informative head-to-head for this idea).
  - `p031_legal_move_laplacian_resolvent` (A3 head-form control
    with the occupancy-blocked piece-typed adjacency, per-piece-
    type edge weights, pooled feature-vector read-out, and
    trunk-fusion MLPs that the mixer cannot use).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. The per-square content-
    weight MLP has no direct access to "which square holds an enemy
    piece"; it must rediscover that from whatever piece-occupancy
    structure the trunk has already encoded into the channel
    features. If the trunk under-encodes piece occupancy in early
    blocks, the per-square scalar cannot learn content-dependent
    edge weighting in those blocks; the operator degenerates to a
    multi-hop diffusion over the static chess-rule reach geometry.
  - The static adjacency does not include occupancy-based blocker
    resolution. Rooks "see" through their own piece on the same
    file; bishops "see" through their own piece on the same
    diagonal. If A3 (the primitive used as an additive head with the
    occupancy-blocked piece-typed adjacency) strictly beats this
    mixer, the absence of occupancy blocking is the load-bearing
    cost of the adaptation.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The sequential K-step matmul inside the mixer is the only
    sequential dependency in the BT4 block; if throughput on
    matched hardware falls below ~40% of the `conv` baseline,
    the matched comparison must drop `model.num_blocks` or
    `model.channels` to compensate (see `trainer_notes.md`).

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
  multi-hop tactical slices where the chess-rule legal-move
  Laplacian + Neumann closure can plausibly help (pin motifs, skewer
  motifs, discovered-attack motifs, battery motifs on files /
  diagonals / ranks, X-ray attacks on the king); slices with sliding
  pieces dominating the position (rook-on-open-file, bishop-pair-
  on-long-diagonal, queen-on-open-line); and the mid-to-upper
  `crtk_difficulty` band where the trunk's local-receptive-field
  stack and the dense `attention` baseline are both likely to be
  insufficient (the conv stack misses multi-hop tactical chains;
  dense attention has no chess-rule legal-move prior).
- Slices where this idea is expected to fail or merely match:
  local-tactical slices (one- or two-square forks, simple captures,
  knight-tactic motifs) where the conv mixer's local 3x3 window
  already saturates; positions where sliding pieces are absent or
  pinned (most pieces blocked at distance 1 by the static reach
  geometry mis-counting occupied squares -- early `crtk_phase`
  openings with many own-color blockers); the lowest
  `crtk_difficulty` band where the trunk's exchange features
  already saturate. These should be measured for non-regression,
  not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer A5 `k1_gat_rebrand` and A6
  `zero_alpha` ablations. If any of these matches this idea on the
  target slice, the multi-hop Neumann-series prior is not load-
  bearing inside the BT4 tower. A2 (`bt4_attention_mixer` -- dense
  all-pairs mixing without the chess-rule legal-move prior) is the
  canonical falsifier: if dense attention matches, the chess-rule
  legal-move Laplacian + multi-hop closure is just a constrained
  form of attention. A8 (`shuffle_adjacency`) closing the slice
  also kills the chess-rule reach prior claim.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
