# i005 Null-Move Contrast Benchmark

## What I Did

- Ran the next TODO benchmark target: `i005_null_move_contrast_puzzle_network`.
- Used the existing paper-grade idea config at `ideas/registry/i005_null_move_contrast_puzzle_network/config.yaml`.
- Trained on CUDA with mixed precision, TF32, automatic worker selection, pinned memory, and persistent workers.
- Completed the full configured 20 epochs; best validation epoch was epoch 18.
- Regenerated CRTK slice reports for validation and test predictions.
- Validated the saved run artifact set after slice generation.
- Linked the run from `idea.yaml` and updated the generated TODO/index inputs.

## Command Log

```bash
PYTHONUNBUFFERED=1 python ideas/registry/i005_null_move_contrast_puzzle_network/train.py
python scripts/reports/report_prediction_slices.py --run-dir results/20260429_023704_idea_i005_null_move_contrast_simple18 --splits val test
python scripts/validate_run_artifacts.py results/20260429_023704_idea_i005_null_move_contrast_simple18
```

## Run

- Result directory: `results/20260429_023704_idea_i005_null_move_contrast_simple18`
- Run timestamp: `2026-04-29T02:59:20Z`
- Model: `null_move_contrast_puzzle_network`
- Training mode: `puzzle_binary`
- Input encoding: `simple_18`
- Device: `cuda`
- Parameters: `240578`
- Dataset: `data/splits/crtk_sample_3class_unique_crtk_tags`
- Best epoch: `18`
- Validation objective: F1
- Seed: `42`

## Aggregate Metrics

| Split | Accuracy | F1 | Precision | Recall | PR AUC | ROC AUC | Brier | Calibration Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.8493 | 0.7932 | 0.7307 | 0.8674 | 0.8596 | 0.9293 | 0.1085 | 0.0477 |
| test | 0.8479 | 0.7913 | 0.7292 | 0.8649 | 0.8538 | 0.9274 | 0.1101 | 0.0490 |

Test confusion matrix:

```text
true 0 -> pred 0: 25182
true 0 -> pred 1:  4818
true 1 -> pred 0:  2027
true 1 -> pred 1: 12973
```

Fine-label to binary-output matrix on test:

```text
known non-puzzle   -> pred 0: 13657, pred 1: 1343
verified near-puz -> pred 0: 11525, pred 1: 3475
verified puzzle   -> pred 0:  2027, pred 1: 12973
```

## Slice Findings

The model is strongest when the position already has a decisive static evaluation. Test accuracy by eval bucket reaches `0.9491` on `winning_black`, `0.9577` on `winning_white`, `0.9673` on `crushing_white`, and `0.9730` on `crushing_black`.

The weakest test slices are:

| Slice | Rows | Wrong | Accuracy | Main Error Pattern |
| --- | ---: | ---: | ---: | --- |
| `crtk_eval_bucket=equal` | 7376 | 1996 | 0.7294 | high near-puzzle false positives and missed puzzles |
| `crtk_difficulty=hard` | 9053 | 2337 | 0.7419 | mixed false positives and false negatives |
| `crtk_tactic_motifs=mate_in_1` | 2077 | 474 | 0.7718 | high false-positive rate on near-puzzles |
| `crtk_tactic_motifs=promotion` | 1211 | 254 | 0.7903 | high false-negative rate on true puzzles |
| `crtk_tactic_motifs=underpromotion` | 1211 | 254 | 0.7903 | same rows as promotion slice |
| `crtk_tag_families=THREAT` | 1211 | 254 | 0.7903 | same promotion/underpromotion weakness |
| `crtk_difficulty=very_hard` | 12151 | 2446 | 0.7987 | very high near-puzzle false-positive rate |
| `crtk_phase=endgame` | 9032 | 1809 | 0.7997 | weaker near-puzzle separation |

The null-move contrast signal is useful: compared with i009 Tactical Equilibrium, i005 raises test accuracy from `0.8456` to `0.8479`, F1 from `0.7854` to `0.7913`, PR AUC from `0.8469` to `0.8538`, and ROC AUC from `0.9233` to `0.9274`. It also improves over i007 on those aggregate metrics.

The main failure mode is still near-puzzle rejection. On `crtk_difficulty=very_hard`, fine-label-1 accuracy is only `0.3850` and the false-positive rate is `0.6258`. Easy and very-easy rows are also conservative for true puzzles, with positive recall only `0.5623` and `0.4218`, respectively, which suggests the default threshold misses quieter verified puzzles even when the position looks broadly simple.

## Training Behavior

The run was GPU-backed and fast relative to the LC0 BT4 relation models. The final config used CUDA mixed precision, TF32, `batch_size: 128`, automatic worker selection, persistent workers, pinned memory, and `prefetch_factor: 2`. GPU memory use was roughly 1.3 GB on the RTX 4070 Laptop GPU during monitoring.

Validation F1 improved from `0.6938` at epoch 1 to `0.7932` at epoch 18. Epochs 19 and 20 did not beat the selected checkpoint. PR AUC peaked near the same late-epoch window, so the model was still learning useful ranking structure after the minimum 10-epoch reliability floor.

## Artifacts

- Summary: `results/20260429_023704_idea_i005_null_move_contrast_simple18/run_summary.md`
- Final metrics: `results/20260429_023704_idea_i005_null_move_contrast_simple18/metrics_final.json`
- Validation slice report: `results/20260429_023704_idea_i005_null_move_contrast_simple18/slice_report_val.md`
- Test slice report: `results/20260429_023704_idea_i005_null_move_contrast_simple18/slice_report_test.md`
- Validation tagged predictions: `results/20260429_023704_idea_i005_null_move_contrast_simple18/predictions_val_crtk_tags.parquet`
- Test tagged predictions: `results/20260429_023704_idea_i005_null_move_contrast_simple18/predictions_test_crtk_tags.parquet`
- Best checkpoint: `results/20260429_023704_idea_i005_null_move_contrast_simple18/checkpoint_best.pt`
- Last checkpoint: `results/20260429_023704_idea_i005_null_move_contrast_simple18/checkpoint_last.pt`

Artifact validation passed:

```text
OK: results/20260429_023704_idea_i005_null_move_contrast_simple18
```

## Publication Readiness

This is a valid single-seed benchmark with complete artifacts and CRTK slice diagnostics. It is not enough by itself for a research-paper claim. Before publication-level conclusions, repeat with multiple seeds, rerun or confirm the LC0 BT4 baseline on the same canonical tagged split, compare against i007/i009/i013 under matched threshold policy, and report near-puzzle false positives separately from aggregate puzzle-binary metrics.
