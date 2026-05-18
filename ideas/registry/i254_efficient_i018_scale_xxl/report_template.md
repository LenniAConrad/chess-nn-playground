# Report Template

## Run

- Result path:
- Config:
- Cell (capacity / depth / stalk / restriction / execution):
- Seeds (typically 42 / 43 / 44):
- GPU:
- Training budget: `epochs=20`, `min_epochs=10`,
  `early_stopping_patience=5`, `monitor=pr_auc`, `batch_size=128`,
  `reduce_on_plateau (factor=0.5, patience=2)`
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`
- Paired i018 baseline path (same split, same seeds, base scale):
- Paired i018 scale_xl baseline path (the capacity falsifier reference cell):
- Paired stalk-12 diagnostic path (if run):

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC (paper-grade primary):
- Calibration:

## Scale-XXL Diagnostics

- Capacity cell (`channels` x `hidden_dim` x `depth` x `stalk_dim`):
- Restriction mode (`full` / `grouped_lowrank`):
- Restriction rank (if grouped):
- Relation groups (if grouped):
- compile_model / fuse_incidence (must be `false` in the capacity
  branch unless parity has been demonstrated):
- Total parameters (must match the matched-budget table:
  full first XXL 785,217; capacity falsifier 474,437 at scale_xl):

## Packet Diagnostics

- Mechanism family: `sheaf` (inherited from i018)
- Sheaf tension / transport imbalance / per-relation energy /
  triad-defect / pin pressure / king-ring summaries:
- Near-puzzle false positives at matched recall:

## Falsifier Outcomes

- Scale falsifier (P2 - P1 mean PR-AUC; must be at least `+0.003`):
- Geometry falsifier (P2 - F1 mean PR-AUC; must be at least `0.02`):
- Stalk falsifier (best `s = 8` row - best `s > 8` row; non-negative
  means stalk scaling is unsupported in this study):
- Grouped-map falsifier (F3 - P2 mean PR-AUC; outside seed noise
  means grouped low-rank should not replace full maps at `s = 8`):
- Systems falsifier (E1 compile-only A/B; meaningful speedup means
  Python overhead was the bottleneck, not the relation algebra):
- Parity falsifier (any execution-branch cell that fails parity must
  be dropped from speed benchmarks):

## Slice Findings

Summarise performance by, with paired i018 and paired i018 scale_xl
baselines as columns:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

Per-slice false positives for fine label `1`, per-slice false
negatives for fine label `2`, confidence/calibration by slice, and
the highest-confidence wrong examples (FEN, difficulty, phase, motifs)
are all required by `ideas/docs/BENCHMARK_REPORTING.md`.

## Keep / Drop Decision

- [ ] P2 capacity run beats P1 capacity falsifier by at least
      `+0.003` mean PR-AUC (the scale falsifier).
- [ ] F1 geometry falsifier drops PR-AUC by at least `0.02` versus
      intact P2 (i018's load-bearing-geometry result still holds at
      XXL scale).
- [ ] F2 i018-weight transfer produces zero logit diff (correctness
      check; the test suite asserts this hard).
- [ ] No execution-branch cell is promoted unless it passes the
      train-mode mixed-precision parity ladder.
- [ ] Effect persists across seeds 42 / 43 / 44.

If any box fails the corresponding row, do not promote that cell;
keep i018 scale_xl as the canonical i018 family winner until a
different structural change is tested.

## Conclusions

- Does i018 still have profitable headroom when extra parameters go
  into width and head? (yes / no)
- Did the stalk diagnostic clear the stalk falsifier? (yes / no /
  not run)
- Did the grouped low-rank restriction mode clear seed noise?
  (yes / no / not run)
- Was compile-only the win? (yes / no / not run)
- Highest-confidence wrong examples on the winning row (FEN,
  difficulty, phase, motifs):
- Recommended next step (promote i254 first XXL as the new i018 scale
  ladder winner, run the stalk diagnostic, walk the execution branch
  through the parity ladder, or drop the XXL conclusion and try a
  structural change instead):
