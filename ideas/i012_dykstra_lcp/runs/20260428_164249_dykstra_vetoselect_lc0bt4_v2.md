# Dykstra + VetoSelect LC0 BT4 Run

Run: `results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2`
Date: 2026-04-28 UTC / 2026-04-29 Asia-Shanghai
Device: NVIDIA CUDA
Model: `dykstra_vetoselect`
Parameters: 531,584
Task: `puzzle_binary`
Input: board-only `lc0_bt4_112`; no engine scores, PVs, best moves, source labels, verification metadata, or fine-label training target.

## What Was Implemented

This is the second Dykstra-line implementation and the direct follow-up to the A0 audit. It keeps the Soft-Dykstra Latent Constraint Projector but wraps it in a VetoSelect-style evidence model.

The model uses an LC0 BT4-style residual trunk, predicts role/relation/motif/slack variables, runs a four-cycle unrolled Dykstra-style projector, and then emits:

- accepted puzzle probability;
- rejected-positive-evidence probability;
- non-puzzle probability;
- raw puzzle and selector logits;
- projection distance, residual, slack, role, relation, correction, and motif diagnostics.

The loss is a three-action VetoSelect objective with Dykstra-aware hard-negative mining. Non-puzzle rows become stronger decoys when they have high raw puzzle evidence, high CRTK rule texture, low projection distance, and low solver trace residual. The loss also keeps a small anchor BCE term and projection/residual shaping terms.

## Overall Metrics

Best checkpoint by validation F1: epoch 3.

| Split | Accuracy | Precision | Recall | F1 | PR AUC | ROC AUC | Brier | Calibration Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.8344 | 0.6984 | 0.8859 | 0.7811 | 0.8540 | 0.9251 | 0.1146 | 0.0116 |
| Test | 0.8335 | 0.6976 | 0.8834 | 0.7796 | 0.8474 | 0.9230 | 0.1156 | 0.0119 |

Test confusion matrix at default threshold 0.5:

```text
binary rows=true, cols=pred
non-puzzle: [24255,  5745]
puzzle:     [ 1749, 13251]
```

Fine-label to binary matrix at default threshold 0.5:

```text
rows = [known non-puzzle, verified near-puzzle, verified puzzle]
cols = [pred non-puzzle, pred puzzle]
[13373, 1627]
[10882, 4118]
[ 1749, 13251]
```

## Operating Points

The default threshold is intentionally recall-heavy. Validation thresholding gives better deployable operating points:

| Operating Point | Threshold | Accuracy | Precision | Recall | F1 | Total FP | Known FP | Near-Puzzle FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default 0.5 | 0.5000 | 0.8335 | 0.6976 | 0.8834 | 0.7796 | 5745 | 1627 | 4118 |
| val-best F1 | 0.6124 | 0.8471 | 0.7452 | 0.8225 | 0.7820 | 4219 | 1217 | 3002 |
| matched recall 0.80 | 0.6507 | 0.8491 | 0.7645 | 0.7909 | 0.7775 | 3655 | 1065 | 2590 |
| matched recall 0.85 | 0.5753 | 0.8440 | 0.7291 | 0.8465 | 0.7835 | 4717 | 1343 | 3374 |

## Comparison To Current References

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test PR AUC | Test ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LC0 BT4 baseline | 0.8350 | 0.7116 | 0.8490 | 0.7742 | 0.8383 | 0.9170 |
| VetoSelect v2 A3 texture | 0.8469 | 0.7518 | 0.8069 | 0.7784 | 0.8396 | 0.9198 |
| Dykstra-LCP A0 | 0.8190 | 0.6766 | 0.8753 | 0.7632 | 0.8331 | 0.9129 |
| Dykstra + VetoSelect v2 | 0.8335 | 0.6976 | 0.8834 | 0.7796 | 0.8474 | 0.9230 |

Matched-recall false-positive comparison using each run's validation threshold:

| Target Recall | Model | Test Recall | Precision | Total FP | Known Non-Puzzle FP | Near-Puzzle FP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.80 | LC0 BT4 baseline | 0.7919 | 0.7496 | 3968 | 1149 | 2819 |
| 0.80 | VetoSelect v2 A3 texture | 0.7941 | 0.7596 | 3769 | 1016 | 2753 |
| 0.80 | Dykstra-LCP A0 | 0.7873 | 0.7396 | 4158 | 1194 | 2964 |
| 0.80 | Dykstra + VetoSelect v2 | 0.7909 | 0.7645 | 3655 | 1065 | 2590 |
| 0.85 | LC0 BT4 baseline | 0.8465 | 0.7133 | 5104 | 1457 | 3647 |
| 0.85 | VetoSelect v2 A3 texture | 0.8455 | 0.7223 | 4876 | 1303 | 3573 |
| 0.85 | Dykstra-LCP A0 | 0.8445 | 0.7031 | 5349 | 1525 | 3824 |
| 0.85 | Dykstra + VetoSelect v2 | 0.8465 | 0.7291 | 4717 | 1343 | 3374 |

Result: this is the best tested run so far for PR AUC, ROC AUC, F1, and matched-recall false-positive control. It reduces near-puzzle false positives versus VetoSelect v2 by 163 at target recall 0.80 and by 199 at target recall 0.85. The tradeoff is lower default-threshold accuracy and precision, so threshold selection should be treated as part of deployment.

## What The Diagnostics Identify

The Dykstra diagnostics are stronger than in A0. True puzzles are now much closer to the learned feasible set than both near-puzzle and known non-puzzle rows.

| Fine Label | Meaning | Mean Prob Puzzle | Projection Distance | Trace Residual | Slack Mean | Role Mass | Relation Mass | Motif Entropy | Reject Logit |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | known non-puzzle | 0.1491 | 0.0640 | 0.0096 | 0.2291 | 15.0593 | 2.5391 | 1.4227 | -6.1054 |
| 1 | verified near-puzzle | 0.3127 | 0.0404 | 0.0056 | 0.2194 | 11.1147 | 2.0414 | 1.3309 | -4.1108 |
| 2 | verified puzzle | 0.7856 | 0.0182 | 0.0023 | 0.0812 | 6.0363 | 1.4250 | 0.2643 | -1.9628 |

Standalone diagnostic AUCs on test:

| Score | Binary Puzzle ROC AUC | Near-vs-Puzzle ROC AUC | Direction |
| --- | ---: | ---: | --- |
| `prob_1` | 0.9230 | 0.8943 | higher means puzzle |
| `reject_positive_logit` | 0.8558 | 0.7976 | higher means accepted puzzle |
| `-projection_distance` | 0.8179 | 0.7620 | lower distance means puzzle |
| `-trace_residual` | 0.8178 | 0.7618 | lower residual means puzzle |
| `-motif_entropy` | 0.8849 | 0.8723 | lower entropy means puzzle |
| `-slack_mean` | 0.8742 | 0.8579 | lower slack means puzzle |

Interpretation: the hybrid objective made the solver trace meaningfully sharper. The projector alone is still not enough to classify, but it now provides useful ordering and helps the VetoSelect head reduce hard-negative false positives at matched recall.

One weakness remains: the explicit `prob_rejected_evidence` head is small for all groups (`0.0123` known non-puzzle, `0.0270` near-puzzle, `0.0252` puzzle). The model mostly expresses rejection through lower accepted-puzzle probability and the reject logit rather than assigning much probability mass to the rejected-evidence class.

## Worst Slices

Worst test slices from the generated CRTK slice report:

| Slice | Rows | Accuracy | FPR | FNR | Near-Puzzle Accuracy | Puzzle Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crtk_eval_bucket=equal` | 7376 | 0.7034 | 0.4610 | 0.1248 | 0.5276 | 0.8752 |
| `crtk_difficulty=hard` | 9053 | 0.7139 | 0.3903 | 0.1377 | 0.5852 | 0.8623 |
| `crtk_tactic_motifs=mate_in_1` | 2077 | 0.7419 | 0.3590 | 0.0587 | 0.5512 | 0.9413 |
| `crtk_tactic_motifs=promotion` | 1211 | 0.7713 | 0.2111 | 0.2878 | 0.7213 | 0.7122 |
| `crtk_tactic_motifs=underpromotion` | 1211 | 0.7713 | 0.2111 | 0.2878 | 0.7213 | 0.7122 |
| `crtk_tag_families=THREAT` | 1211 | 0.7713 | 0.2111 | 0.2878 | 0.7213 | 0.7122 |
| `crtk_eval_bucket=slight_white` | 7085 | 0.7790 | 0.2941 | 0.1117 | 0.6603 | 0.8883 |
| `crtk_eval_bucket=slight_black` | 7378 | 0.7825 | 0.2780 | 0.1184 | 0.6792 | 0.8816 |

The same central failure remains: equal and hard positions still generate too many near-puzzle false positives. The hybrid improves the count, but does not solve that class. Promotions and underpromotions are a different failure mode: the model misses many true puzzles there.

## Decision

Status: tested and currently the strongest benchmark result in this workspace for the high-recall puzzle-binary objective.

Promote Dykstra + VetoSelect v2 as the current best Dykstra-line variant and a serious replacement candidate for VetoSelect v2 when an operating threshold is selected from validation. Do not use the raw 0.5 threshold as the headline deployment point.

The next improvement should focus on making the rejected-evidence action do more work. Promising changes:

- stronger near-puzzle decoy target or class-1 auxiliary ordering loss;
- temperature/weight tuning so rejected-evidence probability rises on near-puzzles without hurting true-puzzle recall;
- a validation-selected threshold stored alongside the checkpoint for the intended recall target;
- targeted augmentation or auxiliary loss for promotion and underpromotion misses.

## Verification

- `python scripts/validate_run_artifacts.py results/20260428_164249_idea_i012_dykstra_vetoselect_lc0bt4_v2`: passed
- Training completed on CUDA and saved full artifacts.
- `predictions_val.parquet`, `predictions_test.parquet`, CRTK-tagged predictions, slice reports, plots, checkpoints, and resolved config are all present in the run directory.

