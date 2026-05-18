# Report Template

## Run

- Result path:
- Config: `ideas/registry/i259_i018_bt4_ensemble_compression/config.yaml`
- Config variant:
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`
- Paired baselines:
  - `lc0_bt4_classifier` (student-only reference)
  - `oriented_tactical_sheaf_laplacian` (i018 teacher reference)
  - Teacher ensemble (logit-average) computed offline
- Source research packet: `ideas/research/packets/classic/i259_i018_bt4_ensemble_compression.md`

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC (overall):
- Brier score:
- Log loss / NLL:
- ECE / adaptive ECE / class-conditional ECE:
- Calibration: temperature-scaled (`T_s`, `T_b`, `T_d` values, reliability diagram link)

## Distillation Diagnostics

- `teacher_mode` used at training time:
- `fusion_mode`:
- `teacher_alpha`, `teacher_temperature`:
- `diagnostic_hint_keys`:
- Mean `teacher_disagreement` on fine label `1`:
- Mean `teacher_entropy` on fine label `1`:
- Mean `teacher_alpha` on fine label `1` (uncertainty_gated only):
- Mean `diagnostic_hint_<name>` vs `teacher_i018_<name>` Pearson r per key:
- Falsifier ablation (`shuffle_teacher_logits`) matched-recall near-puzzle FP delta:

## Matched-Recall Near-Puzzle FP (primary)

- FP@recall=0.80 vs `lc0_bt4_classifier` baseline:
- FP@recall=0.85 vs `lc0_bt4_classifier` baseline:
- FP@recall=0.80 vs `oriented_tactical_sheaf_laplacian` baseline:
- FP@recall=0.85 vs `oriented_tactical_sheaf_laplacian` baseline:
- FP@recall=0.80 vs offline teacher ensemble:

## Slice Findings

Summarize performance by, with `lc0_bt4_classifier` as the paired
baseline column:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

Per-slice false positives for fine label `1`, per-slice false
negatives for fine label `2`, confidence/calibration by slice, and
the highest-confidence wrong examples (FEN, difficulty, phase,
motifs) are all required by `ideas/docs/BENCHMARK_REPORTING.md`.

## Keep / Drop Decision

- [ ] Distillation falsifier (`shuffle_teacher_logits`) loses most of the lift
- [ ] Matched-recall near-puzzle FP at recall 0.80 strictly better than `lc0_bt4_classifier`
- [ ] Matched-recall near-puzzle FP at recall 0.85 strictly better than `lc0_bt4_classifier`
- [ ] Aggregate test PR-AUC within one seed-to-seed std of the best teacher
- [ ] Calibration metrics (ECE, Brier) match or beat `lc0_bt4_classifier` after temperature scaling
- [ ] Student latency within roughly BT4 batch-1 CPU envelope

If any box fails: do not promote i259 over `lc0_bt4_classifier`.

## Conclusions

- Did distillation transfer the ensemble boundary on the near-puzzle slice?
- Which fusion mode produced the best Phase B teacher signal?
- Did the diagnostic-hint heads regress to non-trivial targets?
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with different student shape):
