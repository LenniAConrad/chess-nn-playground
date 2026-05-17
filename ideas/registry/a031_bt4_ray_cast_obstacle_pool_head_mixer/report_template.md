# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Ray-cast diagnostics: per-block mean and max of the per-
    direction accumulator norms `||Y_d||` after the geometric
    scan, per-direction learned decay `gamma_d` trajectory across
    training, and the share of the projection-back contribution
    coming from each direction relative to the residual path.
    Flag mixer collapse (all `||Y_d|| ~ 0`) or direction
    degeneracy (one `gamma_d` dominates) even when the model is
    competitive.
  - Occlusion-proxy diagnostics: per-block distribution of the
    learned soft occupancy `O = sigmoid(Conv1x1(X))` -- mean,
    fraction of cells with `O > 0.5`, and per-square correlation
    with the rule-exact `simple_18` occupancy. If the proxy is
    near-uniform or anti-correlated with the rule mask, the
    `drop_occlusion` ablation should match this idea.
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p026_ray_cast_obstacle_pool_head` (A3 head-form control with
    the rule-exact piece-plane occupancy that the mixer cannot
    read).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer reads a generic `(B, C, 8, 8)` channel tensor rather
    than the `simple_18` piece planes. If the soft occupancy
    proxy `O = sigmoid(Conv1x1(X))` fails to recover the rule-exact
    blocker structure, occlusion termination is decorative and the
    geometric series runs over the full ray; report the
    `drop_occlusion` ablation alongside the headline number.
  - Per-direction learned decays `gamma_d` can collapse to a single
    shared value, in which case the per-direction stack
    degenerates to an isotropic decayed pool; inspect block-level
    `gamma_d` before declaring null.
  - The sequential `max_ray_length`-step prefix scan adds wall-
    clock cost that is not amortised by signal. If
    `train_samples_per_second` falls far below the matched conv
    baseline, the matched-capacity comparison must be re-run at
    shrunken `channels` / `num_blocks`.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the per-direction accumulator magnitude is
    small.

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
  long-range tactical slices where ray-pooled per-direction
  influence is load-bearing -- back-rank pressure, pins, x-rays,
  skewers, sliding-piece batteries, queen / rook / bishop
  long-diagonal motifs -- and the mid-to-upper `crtk_difficulty`
  band where the trunk's local-receptive-field stack is most likely
  to be insufficient. Also expected to win on `crtk_phase`
  `middlegame` and `endgame` slices where the line of attack /
  defence depends on the first blocker along a ray rather than on
  a 3x3 local pattern.
- Slices where this idea is expected to fail or merely match:
  short-range tactics (knight forks, single-square hanging-piece
  motifs, simple captures with no blocker geometry), opening-phase
  positions where the long rays are densely blocked, and the
  lowest `crtk_difficulty` band where the trunk's exchange
  features already saturate. These should be measured for non-
  regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the in-mixer `drop_occlusion` ablation
  (force `O := 0`). If either matches this idea on the target
  slice, the per-direction occlusion-terminated geometric series
  is not load-bearing inside the BT4 tower. A6
  (`shuffle_directions`) closing the slice also kills the
  direction-specific-learning claim.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
