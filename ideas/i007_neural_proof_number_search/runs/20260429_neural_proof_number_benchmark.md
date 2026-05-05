# i007 Neural Proof-Number Search Benchmark

## What I Did

- Ran the next TODO benchmark target: `i007_neural_proof_number_search`.
- Used the existing paper-grade idea config at `ideas/i007_neural_proof_number_search/config.yaml`.
- Trained on CUDA with mixed precision and TF32 enabled through the config.
- Completed the full configured 20 epochs; best validation epoch was epoch 20.
- Regenerated CRTK slice reports for validation and test predictions.
- Validated the saved run artifact set.
- Linked the run from `idea.yaml` and updated the generated TODO/index inputs.

## Command Log

```bash
PYTHONUNBUFFERED=1 python ideas/i007_neural_proof_number_search/train.py
python scripts/reports/report_prediction_slices.py --run-dir results/20260428_183322_idea_i007_neural_proof_number_simple18 --splits val test
python scripts/validate_run_artifacts.py results/20260428_183322_idea_i007_neural_proof_number_simple18
```

## Run

- Result directory: `results/20260428_183322_idea_i007_neural_proof_number_simple18`
- Run timestamp: `2026-04-28T19:16:12Z`
- Model: `neural_proof_number_search`
- Training mode: `puzzle_binary`
- Input encoding: `simple_18`
- Device: `cuda`
- Parameters: `275909`
- Dataset: `data/splits/crtk_sample_3class_unique_crtk_tags`
- Best epoch: `20`
- Validation objective: F1

## Aggregate Metrics

| Split | Accuracy | F1 | Precision | Recall | PR AUC | ROC AUC | Brier | Calibration Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.8466 | 0.7831 | 0.7407 | 0.8307 | 0.8458 | 0.9224 | 0.1095 | 0.0366 |
| test | 0.8454 | 0.7811 | 0.7395 | 0.8275 | 0.8399 | 0.9199 | 0.1110 | 0.0384 |

Test confusion matrix:

```text
true 0 -> pred 0: 25628
true 0 -> pred 1:  4372
true 1 -> pred 0:  2587
true 1 -> pred 1: 12413
```

Fine-label to binary-output matrix on test:

```text
known non-puzzle   -> pred 0: 13803, pred 1: 1197
verified near-puz -> pred 0: 11825, pred 1: 3175
verified puzzle   -> pred 0:  2587, pred 1: 12413
```

## Slice Findings

The model is strongest when the position is already tactically decisive. Test accuracy by eval bucket reaches `0.9516` on `winning_black`, `0.9577` on `winning_white`, `0.9697` on `crushing_black`, and `0.9725` on `crushing_white`.

The weakest test slices are:

| Slice | Rows | Wrong | Accuracy | Main Error Pattern |
| --- | ---: | ---: | ---: | --- |
| `crtk_eval_bucket=equal` | 7376 | 1996 | 0.7294 | high near-puzzle false positives and missed puzzles |
| `crtk_difficulty=hard` | 9053 | 2295 | 0.7465 | mixed false positives and false negatives |
| `crtk_tactic_motifs=promotion` | 1211 | 265 | 0.7812 | high false-negative rate on true puzzles |
| `crtk_tactic_motifs=underpromotion` | 1211 | 265 | 0.7812 | same rows as promotion slice |
| `crtk_tactic_motifs=mate_in_1` | 2077 | 453 | 0.7819 | high false-positive rate on near-puzzles |
| `crtk_difficulty=very_hard` | 12151 | 2589 | 0.7869 | high near-puzzle false-positive rate |
| `crtk_phase=endgame` | 9032 | 1874 | 0.7925 | weaker near-puzzle separation |

Compared with the immediately previous i009 Tactical Equilibrium run, i007 is more conservative: it has fewer test false positives (`4372` vs `4668`) but more false negatives (`2587` vs `2282`), producing lower test F1, PR AUC, ROC AUC, and recall. This suggests the learned proof/disproof tree is useful for rejecting some near-puzzles, but the current bounded pseudo-move search misses true puzzle evidence more often than i009.

## Artifacts

- Summary: `results/20260428_183322_idea_i007_neural_proof_number_simple18/run_summary.md`
- Final metrics: `results/20260428_183322_idea_i007_neural_proof_number_simple18/metrics_final.json`
- Validation slice report: `results/20260428_183322_idea_i007_neural_proof_number_simple18/slice_report_val.md`
- Test slice report: `results/20260428_183322_idea_i007_neural_proof_number_simple18/slice_report_test.md`
- Validation tagged predictions: `results/20260428_183322_idea_i007_neural_proof_number_simple18/predictions_val_crtk_tags.parquet`
- Test tagged predictions: `results/20260428_183322_idea_i007_neural_proof_number_simple18/predictions_test_crtk_tags.parquet`
- Best checkpoint: `results/20260428_183322_idea_i007_neural_proof_number_simple18/checkpoint_best.pt`
- Last checkpoint: `results/20260428_183322_idea_i007_neural_proof_number_simple18/checkpoint_last.pt`

Artifact validation passed:

```text
OK: results/20260428_183322_idea_i007_neural_proof_number_simple18
```

## Publication Readiness

This is a valid single-seed benchmark with complete artifacts and CRTK slice diagnostics. It is not enough by itself for a research-paper claim. Before publication-level conclusions, repeat with multiple seeds, compare against i009 under matched recall/precision thresholds, rerun or confirm the LC0 BT4 baseline on the same canonical tagged split, and compare against the current VetoSelect and Dykstra result paths.
