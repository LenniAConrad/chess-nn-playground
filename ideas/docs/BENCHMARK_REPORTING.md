# Benchmark Reporting Standard

Every registered idea must report aggregate performance and slice-level behavior. A single `3x2` fine-label diagnostic matrix is not enough for deciding whether an architecture learned useful chess structure.

## Canonical Data

Use the clean tagged benchmark splits unless a run note explicitly says otherwise:

```text
data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
```

The CRTK columns are metadata for benchmarking and error analysis only. They must not be used as neural-network input features.

## Reliable Training Budget

Follow `docs/reliable_training_protocol.md` when deciding whether a run is evidence or just a smoke/triage check.

Minimum interpretation rules:

- 1 epoch is a smoke test only;
- 3 epochs is triage only;
- a reliable benchmark run uses the canonical tagged split, `device: nvidia`, a convergence budget, at least 10 active epochs before early stopping, and full artifact validation;
- promotion-grade repo claims should use 3 seeds: 42, 43, and 44;
- paper-grade claims require convergence-budget training, matched baselines, repeated seeds, validation-only threshold selection, ablations, slice analysis, and confidence intervals.

## Required Outputs

Every main run and central ablation must produce:

- aggregate validation and test metrics: accuracy, F1 or macro-F1, PR AUC where defined, ROC AUC where defined, calibration summary, and loss;
- the `3x2` fine-label diagnostic for `fine_label 0/1/2 -> predicted non-puzzle/puzzle` for binary runs, or the native `3x3` matrix for fine-3-class runs plus the collapsed `3x2` view;
- `slice_report_val.md` and `slice_report_test.md` from `scripts/reports/report_prediction_slices.py`;
- tagged prediction parquet files, for example `predictions_test_crtk_tags.parquet`;
- highest-confidence wrong examples with FEN, true label, predicted label, confidence, difficulty, phase, and motifs;
- a short "what this model can and cannot learn" section summarizing the strongest and weakest slices.

## Required Slices

Report at minimum:

- `crtk_difficulty`: very_easy, easy, medium, hard, very_hard;
- `crtk_phase`: opening, middlegame, endgame;
- `crtk_eval_bucket`: equal, slight, clear, winning, crushing buckets by side;
- `crtk_tactic_motifs`: fork, pin, skewer, hanging, overload, discovered_attack, mate_in_1, promotion, underpromotion, and `(none)`;
- `crtk_tag_families`: TACTIC, ENDGAME, OUTPOST, THREAT, and broad always-present families for sanity checking;
- fine-label rows inside each important slice, especially false positives on fine label `1` and false negatives on fine label `2`;
- confidence and calibration by slice, not just correctness.

## Comparison Rules

For each idea, compare against the strongest same-input baseline on the same clean tagged split. A model is only interesting if it improves either:

- overall test quality without making hard slices worse, or
- a declared target slice such as `hard`, `very_hard`, `endgame`, `pin`, `overload`, `mate_in_1`, or `promotion`, while holding overall performance within a documented tolerance.

If gains appear only in aggregate but disappear on every meaningful difficulty, phase, and motif slice, treat the idea as probably exploiting dataset composition rather than learning useful chess structure.

## Report Commands

After a run has `predictions_val.parquet` and `predictions_test.parquet`, generate slice reports with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reports/report_prediction_slices.py \
  --run-dir results/<run_dir> \
  --tagged-split-dir data/splits/crtk_sample_3class_unique_crtk_tags \
  --splits val test \
  --min-count 100 \
  --limit 20
```

Before treating data as benchmark-ready, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/audit_benchmark_data.py
```
