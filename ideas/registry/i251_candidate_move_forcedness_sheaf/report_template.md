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

- NPFP at validation-derived recall `0.80` (i251 vs paired i018):
- NPFP at validation-derived recall `0.85` (i251 vs paired i018):

## Packet Diagnostics

- Mechanism family: `sheaf` (inherited from i018)
- Mechanism energy:
- Sheaf tension / transport imbalance / per-relation energy /
  triad-defect / pin pressure / king-ring summaries:

## Candidate-Move Forcedness (i251-specific)

- `candidate_entropy` (mean / std across batches):
- `candidate_top1_mass`:
- `candidate_gap`:
- `candidate_check_mass`:
- `candidate_capture_mass`:
- `candidate_promotion_mass`:
- `candidate_underpromotion_mass`:
- `candidate_pin_mass`:
- `candidate_king_zone_mass`:
- `candidate_overflow_count`:
- `candidate_count`:
- `candidate_gate` (mean / std):
- `candidate_delta_logits` (mean / std):
- Top-k move breakdown per board (kind, source, target, pool weight,
  gives_check, capture, pin, promotion) for:
  - 5 true positives,
  - 5 near-puzzle false positives,
  - 5 false negatives.

## Numerical Equivalence Check vs i018

State once per release of `candidate_move_forcedness_sheaf.py`:

- Shared-weights eval-mode `logits` max abs diff with
  `disable_move_branch: true` (must be `0.0` exactly modulo platform FP
  semantics):
- Shared-weights eval-mode `logits` max abs diff at zero-init with the
  default branch (should be under `1e-5`):

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
- [ ] Mean test PR-AUC is at least `+0.003` over i018 on the same
      scale, OR matched-recall NPFP is reduced by `>= 1%` absolute
      without compensating regressions
- [ ] F3 (`disable_move_branch`) is clearly worse than full i251
      across the same seeds
- [ ] F4 (`flat_move_pool`) is not better than full i251
- [ ] No slice regression on `crtk_eval_bucket = equal`, `crtk_phase`,
      or `crtk_tactic_motifs` beyond aggregate noise
- [ ] `candidate_overflow_count` mean is low or `max_candidates` was
      raised to bring it under 10%

If any required box fails: keep i018 as the canonical entry and do not
promote i251 over it.

## Conclusions

- What the model appears able to learn (vs i018):
- What the model appears unable to learn (vs i018):
- Which forcedness diagnostics moved most across difficulty / phase:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with different scale
  or seeds):
