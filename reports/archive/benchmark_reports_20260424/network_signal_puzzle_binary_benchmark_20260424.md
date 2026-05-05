# Network Signal Puzzle-Binary Benchmark

This is the corrected benchmark: models emit one puzzle logit and train with `BCEWithLogitsLoss`.

Training label mapping:

- source `0` random / known non-puzzle -> target `0`
- source `1` near-puzzle / hard negative -> target `0`
- source `2` verified puzzle -> target `1`

Prediction threshold: sigmoid(logit) >= 0.5 means predicted puzzle.

Data split used: `data/splits/crtk_sample_3class`, balanced by source class: 120k/120k/120k train and 15k/15k/15k validation/test.

Metrics chart: `reports/archive/benchmark_reports_20260424/network_signal_puzzle_binary_benchmark_20260424_metrics.png`
Source-class behavior chart: `reports/archive/benchmark_reports_20260424/network_signal_puzzle_binary_benchmark_20260424_source_rates.png`
Raw comparison CSV: `reports/archive/benchmark_reports_20260424/network_signal_puzzle_binary_benchmark_20260424.csv`

## Ranking

| Rank | Model | Test F1 | Test PR AUC | Test Acc | Precision | Recall | Best Epoch | Seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LC0 BT4 tower | 0.7445 | 0.8068 | 0.8183 | 0.7006 | 0.7943 | 1 | 669.1 |
| 2 | Stockfish-style NNUE | 0.7340 | 0.7982 | 0.7988 | 0.6562 | 0.8327 | 3 | 223.1 |
| 3 | CNN | 0.6823 | 0.7026 | 0.7606 | 0.6117 | 0.7714 | 3 | 327.0 |
| 4 | MLP | 0.6503 | 0.7089 | 0.6930 | 0.5242 | 0.8562 | 2 | 220.0 |

## Source-Class Diagnostic Rates

| Model | Random -> puzzle FP | Near-puzzle -> puzzle FP | Puzzle recall |
| --- | ---: | ---: | ---: |
| LC0 BT4 tower | 0.0917 | 0.2477 | 0.7943 |
| Stockfish-style NNUE | 0.1204 | 0.3158 | 0.8327 |
| CNN | 0.1535 | 0.3361 | 0.7714 |
| MLP | 0.2865 | 0.4907 | 0.8562 |

## Test 3x2 Confusion Matrices

Rows are source classes `[0 random, 1 near-puzzle, 2 puzzle]`; columns are predictions `[0 non-puzzle, 1 puzzle]`.

### LC0 BT4 tower

Run: `results/20260424_154030_bench_signal_lc0_bt4_classifier`

```text
[13624, 1376]
[11284, 3716]
[3086, 11914]
```

3x2 matrix image: `results/20260424_154030_bench_signal_lc0_bt4_classifier/fine_to_binary_confusion_matrix_test.png`
2x2 binary matrix image: `results/20260424_154030_bench_signal_lc0_bt4_classifier/confusion_matrix_test.png`
Training dashboard: `results/20260424_154030_bench_signal_lc0_bt4_classifier/training_dashboard.png`
Best checkpoint: `results/20260424_154030_bench_signal_lc0_bt4_classifier/checkpoint_best.pt`

### Stockfish-style NNUE

Run: `results/20260424_152740_bench_signal_stockfish_style_nnue_simple18`

```text
[13194, 1806]
[10263, 4737]
[2510, 12490]
```

3x2 matrix image: `results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/fine_to_binary_confusion_matrix_test.png`
2x2 binary matrix image: `results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/confusion_matrix_test.png`
Training dashboard: `results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/training_dashboard.png`
Best checkpoint: `results/20260424_152740_bench_signal_stockfish_style_nnue_simple18/checkpoint_best.pt`

### CNN

Run: `results/20260424_153505_bench_signal_cnn_simple18`

```text
[12697, 2303]
[9958, 5042]
[3429, 11571]
```

3x2 matrix image: `results/20260424_153505_bench_signal_cnn_simple18/fine_to_binary_confusion_matrix_test.png`
2x2 binary matrix image: `results/20260424_153505_bench_signal_cnn_simple18/confusion_matrix_test.png`
Training dashboard: `results/20260424_153505_bench_signal_cnn_simple18/training_dashboard.png`
Best checkpoint: `results/20260424_153505_bench_signal_cnn_simple18/checkpoint_best.pt`

### MLP

Run: `results/20260424_153123_bench_signal_mlp_simple18`

```text
[10702, 4298]
[7640, 7360]
[2157, 12843]
```

3x2 matrix image: `results/20260424_153123_bench_signal_mlp_simple18/fine_to_binary_confusion_matrix_test.png`
2x2 binary matrix image: `results/20260424_153123_bench_signal_mlp_simple18/confusion_matrix_test.png`
Training dashboard: `results/20260424_153123_bench_signal_mlp_simple18/training_dashboard.png`
Best checkpoint: `results/20260424_153123_bench_signal_mlp_simple18/checkpoint_best.pt`

## Interpretation

- Accuracy is higher than the previous 3-class run because this is now a binary task with a 2:1 non-puzzle:puzzle class ratio in validation/test.
- The hard-negative row is the key benchmark signal. A model that calls many near-puzzles puzzle-like is not good enough, even if ordinary binary accuracy looks acceptable.
- The data used here is the existing local labeled CRTK sample split with `known_non_puzzle`, `verified_near_puzzle`, and `verified_puzzle` labels. I verified the labels and split counts locally, but did not independently audit the upstream CRTK generation process.
- The BT4 model uses the current FEN-only `lc0_bt4_112` tensor; older history planes are zero until move history is available.
