# Report Template — a016 BT4 Primitive Mixer (legal_edge_compile_scatter)

## Run

- Result path:
- Config: `ideas/registry/a016_bt4_legal_edge_compile_scatter_mixer/config.yaml`
- Seeds:
- GPU:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`

## Aggregate Metrics

- Accuracy / F1 / ROC AUC / PR AUC / Brier / Calibration:
- Per-fine-label diagnostic confusion matrix (rows 0/1/2 -> non-puzzle/
  non-puzzle/puzzle):

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted idea must
attach:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`,
  `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for
  fine label `2`;
- confidence and calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and
  motifs;
- a short conclusion describing what the model appears able and unable
  to learn relative to the conv / attention baselines.

## Architecture-Specific Diagnostics

- Mechanism family: `bt4_mixer`.
- Mixer: `legal_edge_compile_scatter` (geometric typed adjacency +
  per-edge σ-gate + per-type message scatter + LayerNorm).
- Optional probes during eval:
  - per-type mean σ-gate value over the validation set (gate
    saturation indicator);
  - per-type message-projection norm.

## Cross-Mixer Comparison Table

| Mixer | Aggregate PR AUC | Tactical-slice PR AUC | Throughput (samples/s) |
|---|---|---|---|
| `conv` (baseline) | | | |
| `attention` (baseline) | | | |
| `legal_edge_compile_scatter` (this) | | | |

## Slice Findings

- Target slice: positions whose pivotal squares are reached by the
  typed move-pattern adjacency (knight-fork pivots, rook/bishop ray
  endpoints).
- Watch slices: positions where the mixer's geometry adjacency does
  *not* match the deciding piece interaction; lift should not come at
  their expense.
- crtk_difficulty buckets: report PR AUC, FP rate at matched recall,
  and Brier for each bucket.
- crtk_phase buckets: report the same per opening / middlegame /
  endgame split.

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta vs `mixer: conv` >= 0
- [ ] Tactical-slice PR AUC lift vs `mixer: conv` >= 0
- [ ] Wall-clock per epoch <= 2x `mixer: conv`
- [ ] `random_typed_edges` ablation loses material slice lift
- [ ] `no_edge_gate` ablation loses material slice lift

If any check fails: drop the `legal_edge_compile_scatter` mixer for
the BT4 tower and document the result in `KNOWLEDGE.md`.
