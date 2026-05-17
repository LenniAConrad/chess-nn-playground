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
    state norm `||h_d||` (the eight selective-scan per-direction
    states), and per-block per-direction read-out energy
    `||C[d] * h_d||`. Flag mixer collapse (all `||Y|| ~ 0`) or
    directional degeneracy (one direction dominates, or all
    directions cancel via anti-correlated `C[d]` vectors).
  - A/B coefficient diagnostics: per-block per-direction histogram
    of the sigmoid A and B values `A_{i, d, c} = sigma(A_proj(x))`,
    `B_{i, d, c} = sigma(B_proj(x))`. Saturation of A at 1 means
    the scan becomes a plain geometric prefix sum (equivalent to
    `p026` RayPool modulated by B). Saturation of A at 0 means the
    scan reduces to a one-step `B * x` injection (mixer collapse to
    near-local). Saturation of B at 0 collapses the operator to the
    zero map. Report per-direction A and B mean, max, and entropy
    on the held-out test set. If A is saturated at 1 or B is
    saturated at 0 everywhere, the selective state-space prior is
    decorative and A5 (`disable_selective_A`) or A6
    (`disable_selective_B`) should match this idea.
  - Direction-correlation diagnostics: per-block correlation
    between per-direction state energies `||C[d] * h_d||` and the
    rule-exact piece-attack maps from `simple_18` (e.g., correlate
    the rook-direction states with rook attack rays, the
    bishop-direction states with bishop attack rays) on the
    held-out test set. If the mixer is learning the chess-attack
    geometry, the rook-direction and bishop-direction states
    should correlate non-trivially with the rule-exact attack maps
    of those piece types. If they do not, the chess-direction prior
    is decorative and A7 (`no_directional_C`) should match this
    idea.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control -- the same all-pairs
    long-range mixing role with no 8-direction selective state-
    space prior; the most informative head-to-head for this idea).
  - `p030_ray_parallel_ssm_head` (A3 head-form control with the
    original pooled feature-vector read-out and trunk-fusion MLPs
    that the mixer cannot use).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. The A/B projections have no
    direct access to "which square holds an enemy piece"; they
    must rediscover that from whatever piece-occupancy structure
    the trunk has already encoded into the channel features. If
    the trunk under-encodes piece occupancy in early blocks, the
    A and B coefficients cannot learn content-dependent
    retention/injection in those blocks; the scan degenerates to a
    plain weighted prefix sum.
  - `C` is per-direction-only, not per-square. The spec's full
    form `y = sum_d C_{i, d} h_{i, d}` would require an additional
    64-row C table per direction. If A3 (the primitive used as a
    pooled head) strictly beats this mixer, the per-direction-only
    `C` simplification is part of why the signal does not survive
    in the mixer adaptation.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream; report per-block routed
    output norm statistics alongside the headline number.
  - The sequential 7-step scan inside the mixer is the only
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
  long-range tactical slices where per-channel selective state-space
  mixing along chess rays is load-bearing (pin motifs, skewer
  motifs, discovered-attack motifs, battery motifs on files /
  diagonals / ranks, X-ray attacks on the king); slices with sliding
  pieces dominating the position (rook-on-open-file, bishop-pair-
  on-long-diagonal, queen-on-open-line); and the mid-to-upper
  `crtk_difficulty` band where the trunk's local-receptive-field
  stack and the dense `attention` baseline are both likely to be
  insufficient (the conv stack misses long rays; dense attention
  has no chess-ray prior).
- Slices where this idea is expected to fail or merely match:
  local-tactical slices (one- or two-square forks, simple captures,
  knight-tactic motifs) where the conv mixer's local 3x3 window
  already saturates; positions with very short rays (most pieces
  blocked at distance 1 -- early `crtk_phase` openings); the
  lowest `crtk_difficulty` band where the trunk's exchange
  features already saturate. These should be measured for
  non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer A5 `disable_selective_A` and
  A6 `disable_selective_B` ablations. If any of these matches this
  idea on the target slice, the selective state-space prior is not
  load-bearing inside the BT4 tower. A2 (`bt4_attention_mixer` --
  dense all-pairs mixing without the 8-direction selective-scan
  prior) is the canonical falsifier: if dense attention matches,
  the 8-direction selective state-space scan is just a cheaper
  variant of attention. A9 (`ray_length=1`) closing the slice also
  kills the long-range scan claim.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
