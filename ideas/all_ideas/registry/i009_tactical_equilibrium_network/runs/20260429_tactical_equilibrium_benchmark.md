# i009 Tactical Equilibrium Benchmark

## What I Did

- Ran the next TODO benchmark target: `i009_tactical_equilibrium_network`.
- Used the existing paper-grade idea config at `ideas/all_ideas/registry/i009_tactical_equilibrium_network/config.yaml`.
- Trained on CUDA with mixed precision and TF32 enabled through the config.
- Regenerated CRTK slice reports for validation and test predictions.
- Validated the saved run artifact set.
- Linked the run from `idea.yaml` and updated the generated TODO/index inputs.

## Command Log

```bash
PYTHONUNBUFFERED=1 python ideas/all_ideas/registry/i009_tactical_equilibrium_network/train.py
python scripts/reports/report_prediction_slices.py --run-dir results/20260428_180243_idea_i009_tactical_equilibrium_simple18 --splits val test
python scripts/validate_run_artifacts.py results/20260428_180243_idea_i009_tactical_equilibrium_simple18
```

## Run

- Result directory: `results/20260428_180243_idea_i009_tactical_equilibrium_simple18`
- Run timestamp: `2026-04-28T18:28:04Z`
- Model: `tactical_equilibrium_network`
- Training mode: `puzzle_binary`
- Input encoding: `simple_18`
- Device: `cuda`
- Parameters: `176676`
- Dataset: `data/splits/crtk_sample_3class_unique_crtk_tags`
- Best epoch: `18`
- Validation objective: F1

## Aggregate Metrics

| Split | Accuracy | F1 | Precision | Recall | PR AUC | ROC AUC | Brier | Calibration Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.8486 | 0.7893 | 0.7359 | 0.8510 | 0.8540 | 0.9258 | 0.1089 | 0.0384 |
| test | 0.8456 | 0.7854 | 0.7315 | 0.8479 | 0.8469 | 0.9233 | 0.1102 | 0.0410 |

Test confusion matrix:

```text
true 0 -> pred 0: 25332
true 0 -> pred 1:  4668
true 1 -> pred 0:  2282
true 1 -> pred 1: 12718
```

Fine-label to binary-output matrix on test:

```text
known non-puzzle   -> pred 0: 13688, pred 1:  1312
verified near-puz -> pred 0: 11644, pred 1:  3356
verified puzzle   -> pred 0:  2282, pred 1: 12718
```

## Slice Findings

The model is strongest when the static position already has a decisive evaluation signal. Test accuracy by eval bucket reaches `0.9525` on `winning_black`, `0.9595` on `winning_white`, `0.9752` on `crushing_white`, and `0.9764` on `crushing_black`.

The weakest test slices are:

| Slice | Rows | Wrong | Accuracy | Main Error Pattern |
| --- | ---: | ---: | ---: | --- |
| `crtk_eval_bucket=equal` | 7376 | 1951 | 0.7355 | high near-puzzle false positives and missed puzzles |
| `crtk_difficulty=hard` | 9053 | 2306 | 0.7453 | mixed false positives and false negatives |
| `crtk_tactic_motifs=mate_in_1` | 2077 | 492 | 0.7631 | high false-positive rate on near-puzzles |
| `crtk_tactic_motifs=promotion` | 1211 | 257 | 0.7878 | high false-negative rate on true puzzles |
| `crtk_tactic_motifs=underpromotion` | 1211 | 257 | 0.7878 | same rows as promotion slice |
| `crtk_difficulty=very_hard` | 12151 | 2492 | 0.7949 | high near-puzzle false-positive rate |
| `crtk_phase=endgame` | 9032 | 1840 | 0.7963 | weaker near-puzzle separation |

The model appears to learn a useful tactical-equilibrium signal for puzzlehood, but it still confuses verified near-puzzles with true puzzles in the hardest/equal-eval regimes. It also misses a meaningful fraction of promotion/underpromotion puzzles, where the required tactic may be encoded in a low-material or non-capture move rather than in broad attacker/defender pressure.

## Artifacts

- Summary: `results/20260428_180243_idea_i009_tactical_equilibrium_simple18/run_summary.md`
- Final metrics: `results/20260428_180243_idea_i009_tactical_equilibrium_simple18/metrics_final.json`
- Validation slice report: `results/20260428_180243_idea_i009_tactical_equilibrium_simple18/slice_report_val.md`
- Test slice report: `results/20260428_180243_idea_i009_tactical_equilibrium_simple18/slice_report_test.md`
- Validation tagged predictions: `results/20260428_180243_idea_i009_tactical_equilibrium_simple18/predictions_val_crtk_tags.parquet`
- Test tagged predictions: `results/20260428_180243_idea_i009_tactical_equilibrium_simple18/predictions_test_crtk_tags.parquet`
- Best checkpoint: `results/20260428_180243_idea_i009_tactical_equilibrium_simple18/checkpoint_best.pt`
- Last checkpoint: `results/20260428_180243_idea_i009_tactical_equilibrium_simple18/checkpoint_last.pt`

Artifact validation passed:

```text
OK: results/20260428_180243_idea_i009_tactical_equilibrium_simple18
```

## Publication Readiness

This is a valid single-seed benchmark with complete artifacts and CRTK slice diagnostics. It is not enough by itself for a research-paper claim. Before publication-level conclusions, repeat with multiple seeds, rerun or confirm the LC0 BT4 baseline on the same canonical tagged split, and compare against the current VetoSelect and Dykstra result paths using the same matched-recall false-positive analysis.
