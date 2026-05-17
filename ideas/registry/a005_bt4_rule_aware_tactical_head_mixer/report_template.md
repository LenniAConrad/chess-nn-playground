# Report Template

## Run

- Result path:
- Config: `ideas/registry/a005_bt4_rule_aware_tactical_head_mixer/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an aggregate
confusion matrix. Every promoted idea must report:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for fine
  label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and motifs;
- a short conclusion describing what the model appears able and unable to
  learn.

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration (ECE / Brier):

## Architecture-Specific Diagnostics

- Mechanism family: `bt4_mixer` with `mixer=rule_aware_tactical_head`
- Tower: shared BT4 residual stack from `bt4_primitive_mixer` (stem conv,
  N residual+SE blocks, value head)
- Per-block gated additive fusion:
  `y = base_mix(x) + sigmoid(gate(x)) * delta(forcing(x))`
- Mean / max / `fraction > 0.5` of `sigmoid(gate(x))` on:
  - tactical positives (`crtk_tactic_motifs` non-empty)
  - quiet / non-tactical positions (`crtk_tactic_motifs` empty)
- Distribution of `delta(forcing(x))` magnitude on the same two buckets.
- Per-direction (4 rook + 4 bishop) average response on `crtk_phase`
  slices (opening / middlegame / endgame).

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  - `crtk_tactic_motifs = mate_in_1`
  - `crtk_tactic_motifs` containing checks / captures / promotions
  - high `crtk_difficulty` buckets where forcing geometry matters
- Slices where this idea is expected to fail:
  - `crtk_eval_bucket = equal` quiet positions (the gate should sit near 0)
  - stalemate-vs-checkmate discrimination (rule-exact, not surrogate)
- Ablation that should erase the slice-level gain: `shuffle_tsdp`
  (permute forcing-feature channels) must lose >= 50% of the tactical
  slice lift; `disable_gate` must regress to the conv baseline.
- Minimum useful slice-level improvement: +0.02 PR AUC on
  `crtk_tactic_motifs = mate_in_1` over the conv mixer baseline at matched
  budget.

## Baseline Comparison Table

| Mixer | aggregate PR AUC | mate_in_1 PR AUC | gate mean (tactical) | gate mean (quiet) | step time vs conv |
|---|---|---|---|---|---|
| `conv` (baseline) | | | n/a | n/a | 1.00x |
| `attention` (baseline) | | | n/a | n/a | |
| `rule_aware_tactical_head` (this idea) | | | | | |
| `rule_aware_tactical_head` + `shuffle_tsdp` | | | | | |
| `rule_aware_tactical_head` + `disable_gate` | | | | | |
| `rule_aware_tactical_head` + `zero_delta` | | | | | |

## Keep / Drop Decision

- [ ] aggregate PR AUC delta vs `conv` baseline >= -0.005
- [ ] `crtk_tactic_motifs = mate_in_1` slice lift >= +0.02 PR AUC
- [ ] `shuffle_tsdp` loses >= 50% of the mate_in_1 slice lift
- [ ] `disable_gate` regresses to the conv baseline (within 0.005 PR AUC)
- [ ] `crtk_eval_bucket = equal` slice does not regress
- [ ] Step time within 25% of the conv baseline

If any box fails: drop this mixer from the BT4 mixer family.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, `crtk_difficulty`, `crtk_phase`,
  `crtk_tactic_motifs`):
- Recommended next step (promote / drop / re-run with larger tower):
