# Idea Report Template — a015 BT4 Primitive Mixer (ray_occlusion_semiring_scan)

## Run

- Result path:
- Config: `ideas/registry/a015_bt4_ray_occlusion_semiring_scan_mixer/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration (ECE, MCE):
- Wall-clock per epoch versus `bt4_conv_mixer`:

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an aggregate
confusion matrix. This idea is part of the controlled
`a###_bt4_*_mixer` sweep, so every promoted result must report:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for
  fine label `2`;
- confidence / calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and
  motifs;
- a short conclusion describing what the model appears able and unable
  to learn.

## Sibling Comparison Table

| Mixer | Aggregate PR AUC | CRTK class-1 FP at matched recall | Slider-motif PR AUC | wall-clock / epoch |
|---|---|---|---|---|
| `bt4_conv_mixer` | | | | |
| `bt4_attention_mixer` | | | | |
| `bt4_ray_occlusion_semiring_scan_mixer` | | | | |

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline
  (sliding-piece tactics that depend on first-blocker geometry —
  `crtk_tactic_motifs = pin`, `skewer`, `discovered_attack`, `x_ray`,
  and the `crtk_eval_bucket = equal` slice the source primitive was
  designed for):
- Slices where this idea is expected to fail
  (e.g. `mate_in_1`, promotion-heavy motifs, opening-phase positions
  with no exposed sliders — non-regression only):
- Ablation that should erase the slice-level gain
  (`uniform_transmittance`, `constant_direction`, or `no_step_decay`
  in the primitive folder):
- Minimum useful slice-level improvement: target-slice PR AUC delta
  >= +0.010 vs `bt4_conv_mixer`, with aggregate PR AUC delta in
  `[-0.005, +0.010]`.

## Sweep-Level Keep / Drop

- [ ] Aggregate PR AUC over `bt4_conv_mixer` >= +0.005
- [ ] Lift over `bt4_attention_mixer` >= lift over `bt4_conv_mixer`
- [ ] CRTK class-1 matched-recall FP rate <= `bt4_conv_mixer`
- [ ] Wall-clock per epoch within 1.2x of `bt4_conv_mixer`
- [ ] Primitive `uniform_transmittance` ablation loses the lift

If any box fails: drop `a015` from the architecture-study sweep.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, `crtk_difficulty`,
  `crtk_phase`, motifs):
- Recommended next step (promote / drop / re-run with fused ray-scan
  kernel):
