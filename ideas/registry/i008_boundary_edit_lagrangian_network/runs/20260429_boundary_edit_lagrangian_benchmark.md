# i008 Boundary-Edit Lagrangian Benchmark

## What I Did

- Ran the next TODO benchmark target: `i008_boundary_edit_lagrangian_network`.
- Used the existing paper-grade idea config at `ideas/registry/i008_boundary_edit_lagrangian_network/config.yaml`.
- Trained on CUDA with mixed precision, TF32, automatic worker selection, pinned memory, and persistent workers.
- Completed the full configured 20 epochs; best validation epoch was epoch 20.
- Regenerated CRTK slice reports for validation and test predictions.
- Validated the saved run artifact set after slice generation.
- Linked the run from `idea.yaml` and updated the generated TODO/index inputs.

## Command Log

```bash
PYTHONUNBUFFERED=1 python ideas/registry/i008_boundary_edit_lagrangian_network/train.py
python scripts/reports/report_prediction_slices.py --run-dir results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18 --splits val test
python scripts/validate_run_artifacts.py results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18
```

## Run

- Result directory: `results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18`
- Run timestamp: `2026-04-29T03:30:00Z`
- Model: `boundary_edit_lagrangian_network`
- Training mode: `puzzle_binary`
- Input encoding: `simple_18`
- Device: `cuda`
- Parameters: `173764`
- Dataset: `data/splits/crtk_sample_3class_unique_crtk_tags`
- Best epoch: `20`
- Validation objective: F1
- Seed: `42`

## Aggregate Metrics

| Split | Accuracy | F1 | Precision | Recall | PR AUC | ROC AUC | Brier | Calibration Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.8378 | 0.7788 | 0.7141 | 0.8564 | 0.8430 | 0.9204 | 0.1137 | 0.0333 |
| test | 0.8369 | 0.7769 | 0.7142 | 0.8515 | 0.8375 | 0.9182 | 0.1148 | 0.0340 |

Test confusion matrix:

```text
true 0 -> pred 0: 24889
true 0 -> pred 1:  5111
true 1 -> pred 0:  2227
true 1 -> pred 1: 12773
```

Fine-label to binary-output matrix on test:

```text
known non-puzzle   -> pred 0: 13582, pred 1: 1418
verified near-puz -> pred 0: 11307, pred 1: 3693
verified puzzle   -> pred 0:  2227, pred 1: 12773
```

## Slice Findings

The model is strongest when the static position already has a decisive evaluation signal. Test accuracy by eval bucket reaches `0.9499` on `winning_black`, `0.9595` on `winning_white`, `0.9686` on `crushing_white`, and `0.9730` on `crushing_black`.

The weakest test slices are:

| Slice | Rows | Wrong | Accuracy | Main Error Pattern |
| --- | ---: | ---: | ---: | --- |
| `crtk_eval_bucket=equal` | 7376 | 2056 | 0.7213 | high near-puzzle false positives and missed puzzles |
| `crtk_difficulty=hard` | 9053 | 2399 | 0.7350 | mixed false positives and false negatives |
| `crtk_tactic_motifs=mate_in_1` | 2077 | 522 | 0.7487 | high false-positive rate on near-puzzles |
| `crtk_tactic_motifs=promotion` | 1211 | 263 | 0.7828 | high false-negative rate on true puzzles |
| `crtk_tactic_motifs=underpromotion` | 1211 | 263 | 0.7828 | same rows as promotion slice |
| `crtk_tag_families=THREAT` | 1211 | 263 | 0.7828 | same promotion/underpromotion weakness |
| `crtk_eval_bucket=slight_black` | 7378 | 1595 | 0.7838 | near-puzzle false positives plus missed puzzles |
| `crtk_difficulty=very_hard` | 12151 | 2574 | 0.7882 | very high near-puzzle false-positive rate |

This result is a falsification-leaning benchmark for the current boundary-edit formulation. It underperforms the recent i005, i007, i009, and i013 runs on aggregate test F1, PR AUC, ROC AUC, and accuracy. The learned edit-energy bottleneck does not yet give the desired hard-negative separation: on `crtk_difficulty=very_hard`, fine-label-1 accuracy is only `0.3604` and false-positive rate is `0.6479`.

The model also misses many quiet/simple true puzzles. Positive recall is only `0.4668` on easy positions and `0.3412` on very-easy positions, while very-hard recall is `0.9393`; the default threshold appears biased toward detecting noisy tactical pressure rather than reliable puzzlehood.

## Training Behavior

The run was GPU-backed and fast. The final config used CUDA mixed precision, TF32, `batch_size: 128`, automatic worker selection, persistent workers, pinned memory, and `prefetch_factor: 2`. GPU memory use was roughly 1.3-1.5 GB on the RTX 4070 Laptop GPU during monitoring.

Validation F1 improved from `0.6735` at epoch 1 to `0.7788` at epoch 20, with late-epoch gains after the 10-epoch reliability floor. The model is trainable and stable, but the aggregate and slice results do not justify prioritizing this architecture over i005 or i013 without targeted ablations.

## Artifacts

- Summary: `results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/run_summary.md`
- Final metrics: `results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/metrics_final.json`
- Validation slice report: `results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/slice_report_val.md`
- Test slice report: `results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/slice_report_test.md`
- Validation tagged predictions: `results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/predictions_val_crtk_tags.parquet`
- Test tagged predictions: `results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/predictions_test_crtk_tags.parquet`
- Best checkpoint: `results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/checkpoint_best.pt`
- Last checkpoint: `results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18/checkpoint_last.pt`

Artifact validation passed:

```text
OK: results/20260429_030542_idea_i008_boundary_edit_lagrangian_simple18
```

## Publication Readiness

This is a valid single-seed benchmark with complete artifacts and CRTK slice diagnostics. It is not enough by itself for a research-paper claim. Since the result underperforms stronger candidates, publication work should treat this as evidence against the current boundary-edit bottleneck unless ablations show that edit-basis size, solver steps, or threshold calibration recover hard-negative separation.
