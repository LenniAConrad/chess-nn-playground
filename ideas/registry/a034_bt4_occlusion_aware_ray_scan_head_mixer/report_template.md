# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Selective-scan diagnostics: per-block mean and max of the
    routed-token output norm `||Y||`, per-block per-direction
    output norm `||C_d state_d||` (the eight `Conv2d(C -> C, 1x1)`
    contributions), and per-block per-direction state norm
    `||state_d||`. Flag mixer collapse (all `||Y|| ~ 0`) or
    directional degeneracy (one direction dominates, or all
    directions cancel).
  - Blocker-gate diagnostics: per-block per-direction histogram of
    the sigmoid gate values `g_{i, d} = sigma(Conv2d(C -> 8)(x))`.
    Saturation at 0 means the scan emits zero contribution (mixer
    collapse mode); saturation at 1 means the scan becomes a plain
    geometric prefix sum (RayPool / `p026` mode without occupancy
    gating). Report the per-direction gate mean and gate entropy on
    the held-out test set. If the gate is saturated at 0 or 1
    everywhere, the content-dependent termination is decorative
    and A5 (`disable_blocker_gate`) should match this idea.
  - Direction-correlation diagnostics: per-block correlation
    between per-direction projected outputs `C_d state_d` and the
    rule-exact piece-attack maps from `simple_18` (e.g., correlate
    the rook-direction projections with rook attack rays, the
    bishop-direction projections with bishop attack rays) on the
    held-out test set. If the mixer is learning the chess-attack
    geometry, the rook-direction and bishop-direction projections
    should correlate non-trivially with the rule-exact attack maps
    of those piece types. If they do not, the chess-direction prior
    is decorative and A7 (`shuffle_directions`) should match this
    idea.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control -- the same all-pairs
    long-range mixing role with no 8-direction selective-scan
    prior; the most informative head-to-head for this idea).
  - `p029_occlusion_aware_ray_scan_head` (A3 head-form control
    with the original pooled scalar read-out that the mixer
    cannot use).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. The blocker gate has no
    direct access to "which square holds an enemy piece"; it must
    rediscover that from whatever piece-occupancy structure the
    trunk has already encoded into the channel features. If the
    trunk under-encodes piece occupancy in early blocks, the gate
    cannot learn content-dependent termination in those blocks; the
    scan degenerates to a plain prefix sum and the operator
    collapses to RayPool / `p026`.
  - The fixed-feature gate (computed from raw `x` rather than from
    the running state at each step) cannot terminate the ray after
    *discovering* a blocker mid-scan. The state-dependent termination
    that the source primitive cites as its differentiator from `p026`
    is approximated, not exact. If A3 (the primitive used as a
    pooled head with the source-primitive's read-out) strictly beats
    this mixer, the fixed-feature simplification is part of why the
    signal does not survive in the mixer adaptation.
  - The per-direction read-out is `Conv2d(C -> C, 1x1)` summed
    across the 8 directions; this is denser than the source
    primitive's pooled scalar read-out and may overfit. Report the
    eight per-direction projection norm distribution to detect a
    single dominant direction (the operator reduces to a one-
    direction shift) or a near-uniform direction usage (the
    8-direction prior is decorative and A7 `shuffle_directions`
    should match this idea).
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the routed-token output magnitude is small
    relative to the residual stream.
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
  long-range tactical slices where content-dependent ray termination
  is load-bearing (pin motifs, skewer motifs, discovered-attack
  motifs, battery motifs on files / diagonals / ranks, X-ray
  attacks on the king); slices with sliding pieces dominating the
  position (rook-on-open-file, bishop-pair-on-long-diagonal,
  queen-on-open-line); and the mid-to-upper `crtk_difficulty` band
  where the trunk's local-receptive-field stack and the dense
  `attention` baseline are both likely to be insufficient (the conv
  stack misses long rays; dense attention has no chess-ray prior).
- Slices where this idea is expected to fail or merely match:
  local-tactical slices (one- or two-square forks, simple captures,
  knight-tactic motifs) where the conv mixer's local 3x3 window
  already saturates; positions where sliders are obstructed by
  many friendly pieces (the gate must learn to terminate at
  friendly pieces too, which is a more demanding pattern than
  "terminate at enemy pieces"); opening-phase positions with full
  piece complement where most rays are blocked at distance 1 or 2
  (the iterated scan converges in one step and the long-range
  claim is decorative); and the lowest `crtk_difficulty` band
  where the trunk's exchange features already saturate. These
  should be measured for non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer A5 `disable_blocker_gate`
  ablation. If either matches this idea on the target slice, the
  content-dependent termination prior is not load-bearing inside
  the BT4 tower. A2 (`bt4_attention_mixer` -- dense all-pairs
  mixing without the 8-direction selective-scan prior) is the
  canonical falsifier: if dense attention matches, the
  8-direction selective scan is just a cheaper variant of
  attention. A8 (`ray_length=1`) closing the slice also kills
  the long-range scan claim.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
