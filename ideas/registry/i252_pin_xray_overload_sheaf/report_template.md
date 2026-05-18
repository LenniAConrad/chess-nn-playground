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
- Paired i249 baseline path (same split, seeds, scale):

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Matched-recall near-puzzle false positives

- NPFP at validation-derived recall `0.80` (i252 vs paired i249):
- NPFP at validation-derived recall `0.85` (i252 vs paired i249):

## Packet Diagnostics

- Mechanism family: `sheaf` (inherited from i018)
- Mechanism energy:
- Sheaf tension / transport imbalance / per-relation energy /
  triad-defect / pin pressure / king-ring summaries:

## i252-Specific Pressure Diagnostics

- `xray_pressure` (mean / std across batches):
- `skewer_pressure`:
- `discovered_pressure`:
- `pinned_defender_pressure`:
- `overload_pressure`:

## Slice deltas (i252 vs i249)

Per-slice PR-AUC and per-slice matched-recall NPFP, with the four
target slices flagged:

- `crtk_tactic_motifs = pin`
- `crtk_tactic_motifs = skewer`
- `crtk_tactic_motifs = overload`
- `crtk_tactic_motifs = discovered_attack`

## Slice Findings

Summarize performance by, with i018 and i249 as paired baseline columns:

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

- [ ] F2 (topology scramble) still drops test PR-AUC by `>= 0.02`
- [ ] Overall PR AUC is not worse than i249-fast by more than `0.003`
- [ ] Matched-recall NPFP is reduced at recall `0.80` or `0.85`, OR
      mean PR AUC across `pin`, `skewer`, `overload`,
      `discovered_attack` slices rises by `>= 0.010`
- [ ] F3 (`scramble_new_only`) loses most of the dependency-family lift
- [ ] F4 (`family_collapse`) is clearly worse than full i252
- [ ] No slice regression on `crtk_eval_bucket = equal`, `crtk_phase`,
      or non-target `crtk_tactic_motifs` beyond aggregate noise

If any required box fails: keep i249 as the canonical entry and do not
promote i252 over it.

## Conclusions

- What the model appears able to learn (vs i018 / i249):
- What the model appears unable to learn:
- Which new pressure diagnostic moved most across difficulty / phase:
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / re-run with different scale
  or seeds):
