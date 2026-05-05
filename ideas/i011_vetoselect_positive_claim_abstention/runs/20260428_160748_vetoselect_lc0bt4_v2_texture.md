# Run 20260428_160748 VetoSelect LC0 BT4 v2 Texture

- Result path: `results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture`
- Config: `ideas/i011_vetoselect_positive_claim_abstention/config_v2.yaml`
- Variant: A3, self-mined decoys weighted by deterministic current-board tactical texture
- Device requirement: `device: nvidia`
- Actual device: `cuda`
- Best epoch: `3`

## Aggregate Metrics

| Split | Accuracy | Precision | Recall | F1 | PR AUC |
|---|---:|---:|---:|---:|---:|
| Val | 0.8505 | 0.7565 | 0.8132 | 0.7838 | 0.8513 |
| Test | 0.8469 | 0.7518 | 0.8069 | 0.7784 | 0.8396 |

## Source-Class Test Matrix

Rows are original source classes; columns are predicted non-puzzle and predicted puzzle.

```text
class 0 non-puzzle: 13925, 1075
class 1 near-puzzle: 12080, 2920
class 2 puzzle: 2896, 12104
```

## Baseline And A2 Comparison

| Model | Test F1 | Test PR AUC | Test Recall | Test Precision |
|---|---:|---:|---:|---:|
| LC0 BT4 baseline | 0.7742 | 0.8383 | 0.8490 | 0.7116 |
| VetoSelect A2 | 0.7639 | 0.8324 | 0.8371 | 0.7026 |
| VetoSelect A3 texture | 0.7784 | 0.8396 | 0.8069 | 0.7518 |

At matched validation recall target 0.80, applied to test:

| Model | Test Recall | Precision | Ordinary FP | Near-Puzzle FP | Total FP |
|---|---:|---:|---:|---:|---:|
| LC0 BT4 baseline | 0.7919 | 0.7496 | 1149 | 2819 | 3968 |
| VetoSelect A2 | 0.7918 | 0.7356 | 1269 | 3001 | 4270 |
| VetoSelect A3 texture | 0.7941 | 0.7596 | 1016 | 2753 | 3769 |

At matched validation recall target 0.85, applied to test:

| Model | Test Recall | Precision | Ordinary FP | Near-Puzzle FP | Total FP |
|---|---:|---:|---:|---:|---:|
| LC0 BT4 baseline | 0.8465 | 0.7133 | 1457 | 3647 | 5104 |
| VetoSelect A2 | 0.8447 | 0.6968 | 1607 | 3907 | 5514 |
| VetoSelect A3 texture | 0.8449 | 0.7224 | 1301 | 3570 | 4871 |

## Texture And Veto Diagnostics

Mean deterministic rule texture on test:

| Source fine label | Meaning | Mean texture |
|---:|---|---:|
| 0 | known non-puzzle | 0.3961 |
| 1 | near-puzzle | 0.4294 |
| 2 | verified puzzle | 0.3446 |

Default-threshold puzzle prediction rate by source class:

| Source fine label | Puzzle prediction rate |
|---:|---:|
| 0 | 0.0717 |
| 1 | 0.1947 |
| 2 | 0.8069 |

The v2 texture run improves accepted-puzzle false positives, but it no longer makes rejected-evidence the argmax action as often as A2 did. On test, `prob_rejected_evidence` is present as a diagnostic channel, but the final argmax is effectively between non-puzzle and accepted-puzzle. Treat this as an objective improvement, not as proof that the learned veto state is cleanly separated.

## Slice Findings

See `results/20260428_160748_idea_i011_vetoselect_lc0bt4_v2_texture/slice_report_test.md` for the full benchmark slice report. Main test weaknesses:

- `crtk_eval_bucket=equal`: 0.7202 accuracy over 7376 rows.
- `crtk_difficulty=hard`: 2349 wrong over 9053 rows.
- `crtk_difficulty=very_hard`: high positive recall, but high false-positive pressure on negatives.
- `crtk_tactic_motifs=promotion` and `underpromotion`: 0.6259 positive recall.
- `crtk_tactic_motifs=mate_in_1`: strong puzzle recall at 0.9241, but high near-puzzle false positives.

Strong slices:

- `crtk_eval_bucket=crushing_black`: 0.9742 accuracy.
- `crtk_eval_bucket=crushing_white`: 0.9725 accuracy.
- `crtk_difficulty=very_easy`: 0.9671 accuracy.
- `crtk_difficulty=easy`: 0.9470 accuracy.

## Decision

VetoSelect A3 is the first VetoSelect variant that clears the current LC0 BT4 baseline on the same split by test F1 and PR AUC, and it reduces matched-recall false positives. The margin is small, so do not treat it as a settled architecture win without repeated seeds. If continuing VetoSelect, repeat A3 with seeds and consider tuning for explicit rejected-evidence separation. If moving to a new idea, proceed to Soft-Dykstra or Sparse Relation Pursuit.
