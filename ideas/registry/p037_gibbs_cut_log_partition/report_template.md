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

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Required Slice Analysis

Follow `ideas/docs/BENCHMARK_REPORTING.md`. In addition to the aggregate
metrics, this idea's report must include:

- accuracy, recall, false-positive rate, and confidence by `crtk_difficulty`
  (very_easy, easy, medium, hard, very_hard);
- the same metrics by `crtk_phase` (opening, middlegame, endgame), with
  special attention to endgame fortress positions where the cut log-partition
  is expected to fire;
- `crtk_eval_bucket` rows so we can see whether the lift survives once the
  positions are not visually winning;
- motif rows for `crtk_tactic_motifs` including `pin`, `skewer`,
  `discovered_attack`, `fork`, `hanging`, `overload`, `mate_in_1`,
  `promotion`, and `(none)`;
- `crtk_tag_families` sanity rows (TACTIC, ENDGAME, OUTPOST, THREAT);
- per-slice false positives on fine label `1` and false negatives on fine
  label `2`, with confidence and calibration by slice;
- the highest-confidence wrong examples with FEN, `crtk_difficulty`,
  `crtk_phase`, and motifs;
- a short conclusion describing what the cut log-partition appears able and
  unable to learn relative to the i193 trunk.

## Architecture-Specific Diagnostics

- Mechanism family: `response_constraint`
- Primitive: Gibbs Cut Log-Partition Operator (GCLP)
- `primitive_gate` mean / max / fraction > 0.5 on:
  - Positive samples
  - Negative near-puzzle samples
- `primitive_delta` distribution on the same two buckets
- `gibbs_log_partition_mean` and `_max` distributions
- `gibbs_cut_edge_energy` per-sample mean correlation with king safety

## Slice Findings

- Declared target slice: king-safety / fortress positions
  - Required: p037 unablated >= i193 + 0.03 PR AUC on slice
  - Required: A1 (`shuffle_logpartition`) loses >= 70% of that lift
- Watch slice: late-game endgame fortress positions
- Near-puzzle FP rate at matched recall

## Ablation Comparison Table

| Ablation | slice PR AUC | aggregate PR AUC | gate mean on positives | gate mean on negatives |
|---|---|---|---|---|
| `none` | | | | |
| `shuffle_logpartition` | | | | |
| `uniform_edges` | | | | |
| `uniform_sources` | | | | |
| `zero_delta` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] Declared target slice lift >= +0.03
- [ ] A1 (`shuffle_logpartition`) loses >= 70% of the slice lift
- [ ] A2 (`uniform_edges`) loses >= 40% of the slice lift
- [ ] Throughput drop versus i193 < 25%

If any box fails: drop p037.
