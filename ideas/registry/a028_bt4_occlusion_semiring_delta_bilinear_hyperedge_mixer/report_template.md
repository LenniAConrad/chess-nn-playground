# Idea Report Template

- Extra report sections:
  - Mixer-swap comparison table (this idea vs sibling
    `bt4_*_mixer` ideas and `bt4_conv_mixer`, `bt4_attention_mixer`
    baselines) on both aggregate and target-slice PR AUC.
  - Per-block mixer output norm and effective rank (probe the
    intermediate activations from `BT4PrimitiveMixerNet` blocks
    `0..N`).
  - Occlusion-semiring diagnostics: per-block mean and max of the
    learned soft occupancy `O_s = sigmoid(occ_proj(x_s))` on
    positions with many active pieces vs sparse positions;
    transmittance product `prod_{q < l} (1 - O_{c_{r,q}})` per ray
    averaged across positions to show that the gate actually
    discriminates blocked from unblocked rays. Saturation (every
    square reading `O = 1` or every square reading `O = 0`) should
    be flagged even when the model is competitive.
  - Bilinear-hyperedge diagnostics: per-block norm of the per-pair
    embeddings `||edge_{s,p}||` for each of the 4 opposite-direction
    pairs, ratio `||edge_p|| / (||W_L h_left|| * ||W_R h_right||)`
    (collapses toward 1 when `W_L` and `W_R` learn aligned subspaces
    and the bilinear product carries no more signal than a sum).
  - Cost summary: `train_samples_per_second` and parameter count
    relative to `bt4_conv_mixer` and `bt4_attention_mixer`.
- Required comparisons:
  - `bt4_conv_mixer` (primary A1 control).
  - `bt4_attention_mixer` (A2 control).
  - `p023_occlusion_semiring_delta_bilinear_hyperedge` (A3 head-form
    control with the piece-plane occupancy that the mixer cannot
    read).
  - Capacity-matched `bt4_conv_mixer` (A4 control).
- Known blockers:
  - The mixer's soft-occupancy indicator is learned from generic
    channels rather than read off piece planes. If the indicator
    fails to recover occupancy, the transmittance product is
    contaminated by noise from empty squares and the backward
    recurrence degenerates into a directional unweighted sum of
    `V x`; report the in-mixer `zero_occupancy`/`uniform_occupancy`
    ablation alongside the headline number.
  - The bilinear hyperedge `edge_{s,p} = (W_L h_left) (.)
    (W_R h_right)` can collapse to a sum when the `W_L` and `W_R`
    projections learn aligned subspaces (`left ~ right`); inspect
    block-level `||edge_p||` statistics and the
    `disable_bilinear`-style ablation before declaring null.
  - SqueezeExcite + residual + ReLU may absorb most of the mixer's
    contribution if the mixer's output magnitude is small.
  - The Python loop over `RAY_MAX_LEN = 7` in the backward
    recurrence is unrolled at runtime, not at compile time;
    `torch.compile` or `cudagraph` capture may be required to keep
    the per-block wall-clock cost competitive with the matched
    `conv` baseline at large `num_blocks`.

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
  through-the-line motifs where the occlusion gate and the
  opposite-direction bilinear hyperedge are load-bearing --
  `crtk_tactic_motifs` involving `pin`, `skewer`, `xRayAttack`,
  `discoveredAttack`, and `battery`/`doubleAttack` patterns -- plus
  long-line `mate_in_*` slices and `crtk_difficulty` upper-tail
  positions where a single blocker decides the tactic.
- Slices where this idea is expected to fail or merely match:
  `mate_in_1` with no through-line component, quiet-positional
  `crtk_tactic_motifs`, and opening-`crtk_phase` slices -- these are
  not where ray-based transmittance + bilinear hyperedge should
  dominate and should be measured for non-regression, not for lift.
- Ablation that should erase the slice-level gain: A1 (replace the
  mixer with `conv`) and the source-primitive
  `zero_occupancy`/`disable_bilinear` ablations. If any matches this
  idea on the target slice, either the transmittance gate or the
  bilinear hyperedge is not load-bearing inside the BT4 tower.
- Minimum useful slice-level improvement: target-slice PR AUC delta
  `>= 0.010` vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`, and not strictly dominated by
  `bt4_attention_mixer`.
