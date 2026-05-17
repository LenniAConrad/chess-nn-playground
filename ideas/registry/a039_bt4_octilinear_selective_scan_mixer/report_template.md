# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Selective state-space diagnostics: per-block mean and max of
    the routed-token output norm `||Y||`, per-block per-direction
    output norm `||dir_out_k||` over the eight per-direction
    `_scan_direction` outputs before the 8 * C -> C fuse, and
    per-block per-direction fuse contribution `||W_fuse[:, k*C:(k+1)*C]
    * dir_out_k||` to spot dominant or anti-correlated directions.
    Flag mixer collapse (all `||Y|| ~ 0`) or directional
    degeneracy (one direction dominates so seven of the eight
    slots in the fuse are uninformative, or all directions cancel
    via anti-correlated fuse rows).
  - Selectivity-gate diagnostics: per-block per-direction histogram
    of the sigmoid `A_k = sigma(A_proj[k](x))` values and per-
    block per-direction histogram of the injection coefficient
    `B_k = B_proj[k](x)`. Saturation of `A_k` at 1 means the scan
    becomes a plain non-selective geometric prefix sum of
    `B_k(x_t) * x_t` along the scan path (`fixed_transition`-
    equivalent and A5 should match). Saturation of `A_k` at 0 means
    the iterated state collapses to a single-step `B_k(x_t) * x_t`
    injection with no long-range accumulation. Report per-direction
    `A_k` mean, max, and entropy on the held-out test set; flag
    cases where the per-direction `A_k` distribution is identical
    across all eight directions (degeneracy: the operator is
    learning the same gate everywhere and the chess-direction
    prior is decorative).
  - Direction-correlation diagnostics: per-block correlation
    between per-direction state energies `||dir_out_k||` and the
    rule-exact piece-attack maps from `simple_18` (e.g. correlate
    the file/rank directions with rook attack rays and the
    diagonal directions with bishop attack rays) on the held-out
    test set. If the mixer is learning the chess-attack geometry,
    the file/rank-direction states should correlate non-trivially
    with rook attack rays and the diagonal-direction states with
    bishop attack rays. If they do not, the chess-direction prior
    is decorative and A6 (`single_direction`) should match this
    idea.
  - Scan-path coverage check: confirm that the per-direction
    `scan_paths` buffer covers all 64 squares exactly once per
    direction (cardinal: 8 paths of length 8; diagonal: 15 paths
    of variable length 1..8 with `-1` padding) at construction
    time, and report any direction where the path table reduces
    to fewer than the expected coverage after the `valid =
    path[path >= 0]` filter.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control -- the same all-pairs
    long-range mixing role with no 8-direction selective state-
    space prior; the most informative head-to-head for this idea).
  - `p034_octilinear_selective_scan` (A3 head-form control with
    the original `Linear(13)` piece-plane projection and the
    pooled scalar trunk-fusion path that the mixer cannot use).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. The `A_proj` / `B_proj`
    projections have no direct access to "which square holds an
    enemy piece"; they must rediscover that from whatever piece-
    occupancy structure the trunk has already encoded into the
    channel features. If the trunk under-encodes piece occupancy
    in early blocks, `A_k` / `B_k` cannot learn content-conditioned
    selectivity in those blocks and the scan degenerates to a
    plain weighted prefix sum.
  - The per-direction Python scan loop is the only sequential
    dependency in the BT4 block; the asymptotic Mamba parallel-
    scan win is not realised without a Triton kernel (the source
    primitive flags this). If throughput on matched hardware falls
    below ~40% of the `conv` baseline, the matched comparison
    must drop `model.num_blocks` or `model.channels` to
    compensate (see `trainer_notes.md`).
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The 8 per-direction outputs may collapse to redundant features
    (the OSS decomposition is decorative) if the trunk's channels
    are not separable enough into per-direction signal; flag this
    if A6 (`single_direction`) matches the unablated mixer on the
    declared target slice.

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
  long-line tactical slices where per-channel selective state-space
  mixing along chess rays is load-bearing (pin motifs, skewer
  motifs, discovered-attack motifs, battery motifs on files /
  diagonals / ranks, X-ray attacks on the king); slices with
  sliding pieces dominating the position (rook-on-open-file,
  bishop-pair-on-long-diagonal, queen-on-open-line); long-line
  `mate_in_*` slices where a single sliding-piece blocker decides
  the tactic; and the mid-to-upper `crtk_difficulty` band where the
  trunk's local-receptive-field stack and the dense `attention`
  baseline are both likely to be insufficient (the conv stack
  misses long rays in a single block; dense attention has no
  chess-ray prior).
- Slices where this idea is expected to fail or merely match:
  local-tactical slices (one- or two-square forks, simple captures,
  knight-tactic motifs) where the conv mixer's local 3x3 window
  already saturates; positions with very short scan paths (most
  pieces blocked at distance 1 -- early `crtk_phase` openings);
  the lowest `crtk_difficulty` band where the trunk's exchange
  features already saturate. These should be measured for
  non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer A5 `fixed_transition` and A6
  `single_direction` ablations. If any of these matches this idea
  on the target slice, the selective state-space prior or the 8-
  direction decomposition is not load-bearing inside the BT4 tower.
  A2 (`bt4_attention_mixer` -- dense all-pairs mixing without the
  8-direction selective-scan prior) is the canonical falsifier: if
  dense attention matches, the 8-direction selective state-space
  scan is just a cheaper variant of attention. A7
  (`shuffle_features`) closing the slice also kills the data-
  dependent selectivity claim.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
