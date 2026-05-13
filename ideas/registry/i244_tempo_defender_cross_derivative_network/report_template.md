# Idea Report Template

- Extra report sections:
  - 2-D scatter of `(g_T_norm, max_dd)` per held-out sample, coloured by
    fine label (0/1/2). Expect three clusters if the central hypothesis
    holds: small `g_T_norm` (non-puzzle), large `g_T_norm` with small
    `max_dd` (true puzzle), large `g_T_norm` with large `max_dd` (near-
    puzzle).
  - Saliency entropy histogram and saliency argmax overlap with the
    annotated critical defender on a 50-sample manual eval.
  - Per-slot DeltaDelta histogram (`delta_delta_per_slot`) to show
    whether the spectrum tail or the centre carries the discrimination
    signal.

- Required comparisons:
  - i193 baseline on the matched split/seed/scale/budget.
  - All ablations in `ablations.md` (`main_effects_only`,
    `null_board_perturbation`, `attacker_perturbation`,
    `skip_cross_derivative`, `shared_saliency_uniform`).
  - K-sweep (`topk = 1`, `3`, `5`) at constant `tdcd_channels`.

- Known blockers:
  - Tempo flip is implemented as the channel-12 involution, which is the
    correct involution under the simple_18 contract but may underweight
    side-to-move information versus a model that internalises it. If the
    cross-derivative spectrum is too flat across the dataset, consider
    auxiliary STM loss before changing the involution.
  - Saliency may collapse to the king or to the highest-valued enemy
    piece; the `shared_saliency_uniform` ablation is the first signal.
  - Cost is sensitive to `tdcd_channels` and `topk`; if scout-scale wall
    clock exceeds ~3x i193 the ablation comparisons become unfair.

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an aggregate
confusion matrix. Every promoted idea must require:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for
  fine label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and
  motifs;
- a short conclusion describing what the model appears able and unable
  to learn.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  `crtk_eval_bucket = equal`. Lift expected on this slice is the central
  empirical claim.
- Slices where this idea is expected to fail: `mate_in_1`, promotion, and
  stalemate-trap slices remain TSDP/PFCT territory; TDCD is not expected
  to lift them and should be measured for *non-regression* on those
  slices, not for lift.
- Ablation that should erase the slice-level gain: `main_effects_only`
  (or equivalently `no_mixed_partial`). If this ablation matches the full
  model on the equal slice, the cross-derivative is not load-bearing.
- Minimum useful slice-level improvement: equal-slice PR AUC delta
  `>= 0.015` vs i193, with aggregate PR AUC delta in `[-0.005, +0.010]`.
