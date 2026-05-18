# Report Template

## Run

- Result path:
- Config:
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

## Matched-recall near-puzzle false positives

- NPFP at validation-derived recall `0.80` (i250 vs paired i018):
- NPFP at validation-derived recall `0.85` (i250 vs paired i018):

## Packet Diagnostics

- Mechanism family: `sheaf` (inherited from i018)
- Mechanism energy:
- Sheaf tension / transport imbalance / per-relation energy / triad-defect / pin pressure / king-ring summaries:

## Confidence Attribution (i250-specific)

- `confidence_mean`:
- `confidence_max`:
- `confidence_std`:
- `pin_edge_confidence`:
- `king_zone_confidence`:
- Top-k confident edges per board (relation, source, destination, confidence, source piece, target piece, pin/king-zone flag) for:
  - 5 true positives,
  - 5 near-puzzle false positives,
  - 5 false negatives.

## Numerical Equivalence Check vs i018

State once per release of `learned_relation_confidence_sheaf.py`:

- Shared-weights eval-mode `logits` max abs diff with `flat_confidence: true` (must be `0.0` exactly modulo platform FP semantics):
- Shared-weights eval-mode `logits` max abs diff at zero-init with normalized head (should be under `1e-5`):

## Slice Findings

Summarize performance by, with i018 as the paired baseline column:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

Per-slice false positives for fine label `1`, per-slice false negatives
for fine label `2`, confidence/calibration by slice, and the
highest-confidence wrong examples (FEN, difficulty, phase, motifs) are
all required by `ideas/docs/BENCHMARK_REPORTING.md`.

## Keep / Drop Decision

- [ ] Numerical equivalence check vs i018 at zero-init passed
- [ ] F2 (topology scramble) still drops test PR-AUC by `>= 0.02`
- [ ] Mean test PR-AUC is at least `+0.003` over i018 on the same scale, OR matched-recall NPFP is reduced by `>= 1%` absolute without compensating regressions
- [ ] F3 (flat confidence) is clearly worse than full i250 across the same seeds
- [ ] F4 (no normalization) is not better than full i250
- [ ] No slice regression on `crtk_eval_bucket = equal`, `crtk_phase`, or `crtk_tactic_motifs` beyond aggregate noise

If any required box fails: keep i018 as the canonical entry and do not
promote i250 over it.

## Conclusions

- What the model appears able to learn (vs i018):
- What the model appears unable to learn (vs i018):
- Which confidence groups received the largest learned variance:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with different scale or seeds):
