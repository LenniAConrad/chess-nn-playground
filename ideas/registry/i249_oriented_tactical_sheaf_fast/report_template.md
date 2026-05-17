# Report Template

## Run

- Result path:
- Config:
- Config variant: `config.yaml` exact / `config_eval_fp16.yaml` eval-FP16
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`
- Paired i018 baseline path (same split, seeds, scale):

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Speed Diagnostics (i249-specific)

- `train_samples_per_second` (i249 vs paired i018):
- `eval_samples_per_second` (i249 vs paired i018):
- `fit_elapsed_seconds` (i249 vs paired i018):
- Effective speedup (i249 / i018):
- `model.compile_model`:
- `model.compile_mode`:
- `model.return_diagnostics`:
- `model.inference_autocast_dtype`:
- `model.inference_autocast_min_batch`:
- Torch precision (`allow_tf32`, `matmul_precision`):

## Packet Diagnostics

- Mechanism family: `sheaf` (inherited from i018)
- Packet auxiliary logit:
- Mechanism energy:
- Sheaf tension / transport imbalance / per-relation energy / triad-defect / pin pressure / king-ring summaries:
- Near-puzzle false positives:

## Numerical Equivalence Check vs i018

State once per release of the FastSheafDiffusionBlock:

- Shared-weights eval-mode `logits` max abs diff:
- Shared-weights `loss` diff:
- Shared-weights per-parameter gradient max abs diff (`rho_src`, `rho_dst`, `relation_gate_logits`, linears, head):
- Diagnostics diff (`sheaf_tension`, `pin_pressure`, per-relation energies):

These must be within the math_thesis.md tolerances (logits <= 1e-5, grads <= 1e-7). Failing this gate invalidates the entire report.

## Slice Findings

Summarize performance by, with i018 as the paired baseline column:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

Per-slice false positives for fine label `1`, per-slice false negatives
for fine label `2`, confidence/calibration by slice, and the highest-
confidence wrong examples (FEN, difficulty, phase, motifs) are all
required by `ideas/docs/BENCHMARK_REPORTING.md`.

## Keep / Drop Decision

- [ ] Numerical equivalence check vs i018 passed
- [ ] Mean test PR-AUC within one i018 seed-to-seed std (~0.0045) of i018 on every scale
- [ ] Mean `train_samples_per_second` strictly higher than i018 on every scale
- [ ] No slice regression on `crtk_eval_bucket = equal`, `crtk_phase`, or `crtk_tactic_motifs` beyond aggregate noise

If any box fails: keep i018 as the canonical entry and do not promote i249 over it.

## Conclusions

- What the model appears able to learn (vs i018):
- What the model appears unable to learn (vs i018):
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with different compile mode):
