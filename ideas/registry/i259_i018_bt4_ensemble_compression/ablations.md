# Ablations

The source packet's
`Ablations, failure modes, and implementation plan` section calls for
a small, surgical ablation grid. Five of those axes are exposed
directly by this idea's `model.*` knobs and so can be run by simply
sweeping the config without rebuilding the architecture.

## Axes

1. **Student size.** `student_channels`, `student_num_blocks`,
   `student_value_hidden`. Map the BT4 base / scale-up / scale-xl
   shapes the source packet describes.

2. **Fusion mode.** `fusion_mode` in
   `{equal_weight, tuned_alpha, uncertainty_gated}`. Drives the
   `teacher_ensemble_logit` formula.

3. **Teacher temperature / weight.** `teacher_temperature`,
   `teacher_alpha`. Used by Phase A calibration sweeps.

4. **Diagnostic-hint feature set.** `diagnostic_hint_keys`. Subset of
   the six i018 diagnostics the student regresses. Setting the list
   empty disables the diagnostic-hint heads entirely.

5. **Architecture-level ablations.** `model.ablation` in:

   - `none`: full architecture (default).
   - `student_only`: equivalent to `teacher_mode='off'`. Clean
     student baseline.
   - `zero_hint_heads`: zeros the diagnostic-hint outputs so the
     student logit is decoupled from the auxiliary regression.
   - `teacher_logits_only`: rebinds `logits` to
     `teacher_ensemble_logit`. Evaluation-only — do not train.
   - `shuffle_teacher_logits`: in-batch permutation of teacher
     logits. The distillation falsifier — if KD lift survives
     shuffling, the teacher boundary is not load-bearing.

## Required ablation sequence

| Step | Config delta | Pass criterion |
|---|---|---|
| Baseline | default | trains, matches student-only BT4 reference within seed noise |
| Teacher passthrough | `teacher_mode=research`, `ablation=teacher_logits_only` | teacher ensemble logit is finite, calibrated; near-puzzle FP improves over either teacher alone |
| Distillation | `teacher_mode=research`, `ablation=none`, KD/diag lambdas (offline) | matched-recall near-puzzle FP improves over `student_only` baseline at the recall targets |
| Falsifier | `teacher_mode=research`, `ablation=shuffle_teacher_logits` | distillation lift collapses |
| Hint-head ablation | `ablation=zero_hint_heads` | quantifies how much lift is from KD vs the diagnostic-hint regression |
| Fusion mode | `fusion_mode in {equal_weight, tuned_alpha, uncertainty_gated}` | identifies fusion variant with the best Phase B near-puzzle FP |

## Failure modes from the source packet

- **Leakage in learned fusion.** The architecture surface keeps fusion
  stateless (a single linear gate over teacher disagreement / entropy)
  so leakage risk lives in the offline cache pipeline, not the model.
- **Calibration drift.** Always recalibrate per-teacher temperatures
  before generating the cache; the model exposes
  `teacher_temperature` so the calibration step is local to the
  config.
- **Label-noise amplification.** Tracked by the `shuffle_teacher_logits`
  falsifier and by inspecting `teacher_disagreement` densities on the
  fine-label `1` slice.
- **Slice regression.** Audit `crtk_eval_bucket=equal`, hard, mate-in-1,
  promotion, and underpromotion slices on the student's
  `predictions_<split>.parquet`.
- **Shipping the wrong compute shape.** The default config trains a
  pure BT4-shaped student; alternative shapes are reachable through
  `student_*` knobs only, so a "heavy hybrid" student requires a
  deliberate config change.
