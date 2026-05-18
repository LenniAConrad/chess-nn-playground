# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The default config
is paper-grade, CUDA-required, and trains the BT4-shaped conv student
with the shared puzzle_binary BCE-with-logits trainer
(`model.teacher_mode='off'` makes the network behave as a single
student). This keeps the deployment shape that the source packet
explicitly recommends: "one encoding, one student, one calibration
layer".

Differences vs the lc0_bt4_classifier baseline:

- `model.name = i018_bt4_ensemble_compression` (i259 wrapper builder)
- `model.student_*` hyperparameters control the BT4-shaped student
  (named `student_channels`, `student_num_blocks`, etc. so the same
  config can later carry separate teacher hyperparameters without
  collisions).
- `model.teacher_*` and `model.fusion_mode` are inert when
  `teacher_mode='off'` but are validated in the builder so flipping
  the mode does not require schema changes.
- `model.diagnostic_hint_keys` selects which i018 diagnostics the
  student's auxiliary heads regress against during distillation. The
  list is forwarded to the bespoke trunk and surfaces matching
  `diagnostic_hint_<name>` columns in the trainer's prediction
  parquet.

## Loss (default config, teacher_mode='off')

`bce_with_logits` on the student puzzle logit. No KD term is
required for a clean student-only training pass; the model's output
dict still emits zero-filled teacher columns so audits that expect a
stable schema continue to work.

## Phase A through D (research, teacher_mode='research')

The source packet's protocol is:

1. **Phase A** — Calibrate the standalone i018 and BT4 teachers,
   evaluate weighted logit averaging on a held-out fusion subset,
   report matched-recall near-puzzle FP plus calibration. To run this
   with i259: load teacher checkpoints into `teacher_i018` and
   `teacher_bt4` after `build_model_from_config`, set
   `model.teacher_mode='research'`, and dump
   `teacher_ensemble_logit`, `teacher_disagreement`,
   `teacher_entropy`, `teacher_alpha` from
   `predictions_<split>.parquet`.

2. **Phase B** — Generate 5-fold out-of-fold teacher caches, fit the
   linear stacked fuser on OOF predictions only, reserve the
   official validation split for calibration and threshold search.
   The model surfaces the necessary tensors directly; the OOF /
   fuser fit is an offline script.

3. **Phase C** — Generate an offline distillation cache from the
   chosen teacher fusion. Train BT4-shaped students with KD and
   diagnostic-hint heads (`lambda_KD`, `lambda_diag` from the source
   packet's loss equation). Recalibrate each student on the
   validation split.

4. **Phase D** — Single untouched test evaluation with paired
   confidence intervals and the standard report bundle.

## Cost expectation

With `teacher_mode='off'` the model trains at roughly
`lc0_bt4_classifier` speed (one BT4-shaped forward + a handful of
small linear hint heads). With `teacher_mode='research'` the forward
adds one i018 sheaf forward and one extra BT4 forward, both under
`torch.no_grad()`. Expect approximately
`time_off + time_i018 + time_bt4_teacher` wall-clock per step. If
that exceeds the data loader budget, switch to offline cache
generation (run teachers once over the split, persist
`teacher_ensemble_logit` and the hint targets to parquet) and train
the student from that cache.

## Ablation runs

Promotion of i259 requires the distillation falsifier. Use:

```yaml
model:
  ablation: shuffle_teacher_logits
  teacher_mode: research
```

with everything else matched to the unablated run. If the shuffled
run matches the unablated run on matched-recall near-puzzle FP, the
teacher boundary is not load-bearing and the architecture should be
dropped.

Additional ablations to run if the falsifier passes:

- `ablation: student_only` — pure baseline (also reachable via
  `teacher_mode='off'`).
- `ablation: zero_hint_heads` — does the diagnostic-hint regression
  help?
- `ablation: teacher_logits_only` — evaluates the teacher boundary
  directly (do not train with this — teachers are frozen).
- `fusion_mode: equal_weight` vs `tuned_alpha` vs
  `uncertainty_gated` — fusion-capacity ablation from the source
  packet.

## Reports

Standard idea report. Required slices (see `report_template.md`):

- aggregate validation and test PR AUC
- matched-recall near-puzzle FP rate at recall 0.80 / 0.85
- per-slice false positives for fine label `1`
- worst-slice accuracy on the elevated hard slices
- calibration (Brier, log loss, ECE, reliability)

The diagnostic columns `teacher_ensemble_logit`,
`teacher_disagreement`, `teacher_alpha`, and the
`diagnostic_hint_<name>` outputs should be inspected to confirm the
student is regressing toward the teacher boundary instead of
copying noise.
