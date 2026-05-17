# Report Template

## Run

- Result path:
- Config: `ideas/registry/a017_bt4_signed_edit_bilinear_memory_mixer/config.yaml`
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

- Mechanism family: `bt4_mixer` with `mixer=signed_edit_bilinear_memory`
- Tower: shared BT4 residual stack from `bt4_primitive_mixer` (stem conv,
  N residual+SE blocks, value head)
- Per-block SEBM state triple: `s = sum_j a_j`, `u = sum_j b_j`,
  `p = s (.) u - sum_j a_j (.) b_j` over 64 board tokens, followed by FiLM
  broadcast and per-square readout
- Mean / max / `fraction > 0.5` of the FiLM `gamma` magnitude on:
  - positions with high attacker/defender or blocker/slider pair density
    (target slices below)
  - quiet / single-piece positions where pair interactions are sparse
- Per-block bilinear-rank utilisation: fraction of the `bilinear_rank`
  dimensions whose per-batch `(s, u, p)` variance exceeds 5 percent of the
  maximum; tracks rank collapse in the SEBM memory.
- Mean magnitude of the FM cross-term `p` relative to `s (.) u`; if
  `||p|| / ||s (.) u||` collapses to 0 the FM identity is degenerate.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  - `crtk_tactic_motifs` containing attacker/defender or sacrifice motifs
    where pair interactions are load-bearing
  - high `crtk_difficulty` buckets where multi-piece coordination matters
  - `crtk_phase = middlegame` positions with dense pair interactions
- Slices where this idea is expected to fail:
  - `crtk_eval_bucket = equal` quiet positions where the global pair memory
    collapses to a near-constant vector
  - endgame positions with very few active pieces where the FM cross-term
    is dominated by a small number of tokens and rank-1 mixers suffice
- Ablation that should erase the slice-level gain: `shuffle_pair_state`
  (permute the global memory across the batch) must lose >= 50% of the
  target-slice lift; `drop_pair_term` must lose >= 30%; `disable_film`
  must regress to the conv baseline.
- Minimum useful slice-level improvement: +0.02 PR AUC on at least one
  declared target slice over the conv mixer baseline at matched budget.

## Baseline Comparison Table

| Mixer | aggregate PR AUC | target-slice PR AUC | FiLM gamma mean (target) | FiLM gamma mean (quiet) | step time vs conv |
|---|---|---|---|---|---|
| `conv` (baseline) | | | n/a | n/a | 1.00x |
| `attention` (baseline) | | | n/a | n/a | |
| `signed_edit_bilinear_memory` (this idea) | | | | | |
| `signed_edit_bilinear_memory` + `shuffle_pair_state` | | | | | |
| `signed_edit_bilinear_memory` + `drop_pair_term` | | | | | |
| `signed_edit_bilinear_memory` + `disable_film` | | | | | |

## Keep / Drop Decision

- [ ] aggregate PR AUC delta vs `conv` baseline >= -0.005
- [ ] declared target-slice lift >= +0.02 PR AUC over the conv baseline
- [ ] `shuffle_pair_state` loses >= 50% of the target-slice lift
- [ ] `drop_pair_term` loses >= 30% of the target-slice lift
- [ ] `disable_film` regresses to the conv baseline (within 0.005 PR AUC)
- [ ] `crtk_eval_bucket = equal` slice does not regress
- [ ] Step time within 25% of the conv baseline

If any box fails: drop this mixer from the BT4 mixer family.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, `crtk_difficulty`, `crtk_phase`,
  `crtk_tactic_motifs`):
- Recommended next step (promote / drop / re-run with larger
  `bilinear_rank` or larger tower):
