# i013 Sparse Relation Pursuit Asymmetry Benchmark

## What I Did

- Ran the next TODO benchmark target: `i013_sparse_relation_pursuit_asymmetry`.
- Used the existing paper-grade idea config at `ideas/i013_sparse_relation_pursuit_asymmetry/config.yaml`.
- Trained on CUDA with mixed precision, TF32, 8 dataloader workers, pinned memory, and persistent workers.
- Completed the full configured 20 epochs; best validation epoch was epoch 20.
- Regenerated CRTK slice reports for validation and test predictions.
- Validated the saved run artifact set after slice generation.
- Linked the run from `idea.yaml` and updated the generated TODO/index inputs.

## Command Log

```bash
PYTHONUNBUFFERED=1 python ideas/i013_sparse_relation_pursuit_asymmetry/train.py
python scripts/reports/report_prediction_slices.py --run-dir results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4 --splits val test
python scripts/validate_run_artifacts.py results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4
```

## Run

- Result directory: `results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4`
- Run timestamp: `2026-04-29T02:28:14Z`
- Model: `sparse_relation_pursuit_asymmetry`
- Training mode: `puzzle_binary`
- Input encoding: `lc0_bt4_112`
- Device: `cuda`
- Parameters: `121115`
- Dataset: `data/splits/crtk_sample_3class_unique_crtk_tags`
- Best epoch: `20`
- Validation objective: F1
- Seed: `42`

## Aggregate Metrics

| Split | Accuracy | F1 | Precision | Recall | PR AUC | ROC AUC | Brier | Calibration Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.8623 | 0.8096 | 0.7510 | 0.8781 | 0.8748 | 0.9389 | 0.1017 | 0.0574 |
| test | 0.8608 | 0.8073 | 0.7495 | 0.8749 | 0.8687 | 0.9369 | 0.1033 | 0.0581 |

Test confusion matrix:

```text
true 0 -> pred 0: 25614
true 0 -> pred 1:  4386
true 1 -> pred 0:  1877
true 1 -> pred 1: 13123
```

Fine-label to binary-output matrix on test:

```text
known non-puzzle   -> pred 0: 13699, pred 1: 1301
verified near-puz -> pred 0: 11915, pred 1: 3085
verified puzzle   -> pred 0:  1877, pred 1: 13123
```

## Slice Findings

The model is strongest when the static position already has a decisive evaluation signal. Test accuracy by eval bucket reaches `0.9559` on `winning_black`, `0.9674` on `winning_white`, `0.9752` on `crushing_white`, and `0.9764` on `crushing_black`.

The weakest test slices are:

| Slice | Rows | Wrong | Accuracy | Main Error Pattern |
| --- | ---: | ---: | ---: | --- |
| `crtk_eval_bucket=equal` | 7376 | 1792 | 0.7570 | high near-puzzle false positives and missed puzzles |
| `crtk_difficulty=hard` | 9053 | 2118 | 0.7660 | mixed false positives and false negatives |
| `crtk_tactic_motifs=mate_in_1` | 2077 | 425 | 0.7954 | high false-positive rate on near-puzzles |
| `crtk_difficulty=very_hard` | 12151 | 2295 | 0.8111 | very high near-puzzle false-positive rate |
| `crtk_tactic_motifs=promotion` | 1211 | 227 | 0.8126 | high false-negative rate on true puzzles |
| `crtk_tactic_motifs=underpromotion` | 1211 | 227 | 0.8126 | same rows as promotion slice |
| `crtk_tag_families=THREAT` | 1211 | 227 | 0.8126 | same promotion/underpromotion weakness |
| `crtk_eval_bucket=slight_white` | 7085 | 1310 | 0.8151 | near-puzzle false positives plus missed puzzles |

The sparse relation pursuit architecture improves the aggregate benchmark over the two immediately previous runs. Compared with i009 Tactical Equilibrium, i013 raises test accuracy from `0.8456` to `0.8608`, F1 from `0.7854` to `0.8073`, PR AUC from `0.8469` to `0.8687`, and ROC AUC from `0.9233` to `0.9369`. Compared with i007 Neural Proof-Number Search, i013 also improves all four of those aggregate test metrics.

The improvement is not uniformly clean. The model is still too willing to call very-hard near-puzzles positive: on `crtk_difficulty=very_hard`, fine-label-1 accuracy is only `0.4695`, with false-positive rate `0.5520`. It also misses promotion and underpromotion puzzles more often than the aggregate rate, with positive recall `0.7590` on those motif rows.

## Training Behavior

The run was slow but GPU-backed throughout. The final config used CUDA mixed precision, TF32, `batch_size: 64`, `num_workers: 8`, `persistent_workers: true`, `pin_memory: true`, and `prefetch_factor: 2`. GPU memory use was roughly 4.3 GB on an RTX 4070 Laptop GPU, leaving headroom; future speed work should test larger batch sizes or more compact relation-token evaluation.

Validation F1 was non-monotonic: epoch 15 reached `0.8068`, epochs 16-19 traded threshold quality against precision or PR AUC, and epoch 20 finished as the selected checkpoint with validation F1 `0.8096`. Training F1 continued climbing to `0.8599`, so more regularization or threshold calibration may be useful before making publication claims.

## Artifacts

- Summary: `results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/run_summary.md`
- Final metrics: `results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/metrics_final.json`
- Validation slice report: `results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/slice_report_val.md`
- Test slice report: `results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/slice_report_test.md`
- Validation tagged predictions: `results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/predictions_val_crtk_tags.parquet`
- Test tagged predictions: `results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/predictions_test_crtk_tags.parquet`
- Best checkpoint: `results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/checkpoint_best.pt`
- Last checkpoint: `results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4/checkpoint_last.pt`

Artifact validation passed:

```text
OK: results/20260428_192052_idea_i013_sparse_relation_pursuit_lc0bt4
```

## Publication Readiness

This is a valid single-seed benchmark with complete artifacts and CRTK slice diagnostics. It is not enough by itself for a research-paper claim. Before publication-level conclusions, repeat with multiple seeds, rerun or confirm the LC0 BT4 baseline on the same canonical tagged split, compare against i007/i009 under matched threshold policy, and report the near-puzzle false-positive slices separately from aggregate metrics.
