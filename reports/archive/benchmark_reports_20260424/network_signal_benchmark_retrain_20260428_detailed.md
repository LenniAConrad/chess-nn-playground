# Fresh Network Signal Benchmark Report

- Suite report: `reports/experiment_suites/20260428_153502_network_signal_benchmark_suite.json`
- Started: `2026-04-28T15:18:21Z`
- Finished: `2026-04-28T15:35:02Z`
- Status: `ok`
- Device policy: all configs used `device: nvidia`; the retrain ran on PyTorch CUDA.
- Task: `puzzle_binary`, single-logit puzzle detector.

## Label Contract

| Source fine class | Training target | Interpretation |
| --- | --- | --- |
| `0` | `0` | known non-puzzle |
| `1` | `0` | near-puzzle / unresolved candidate, intentionally trained as non-puzzle |
| `2` | `1` | verified puzzle |

The key diagnostic is therefore not just binary accuracy. A good model should keep fine `0` and fine `1` on output `0` while still identifying fine `2` as output `1`. Fine `1` is the hardest negative class because it often contains tactical-looking positions.

## Data Used

| Split | Rows | fine 0 | fine 1 | fine 2 | Binary target 0 | Binary target 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | 360000 | 120000 | 120000 | 120000 | 240000 | 120000 |
| `val` | 45000 | 15000 | 15000 | 15000 | 30000 | 15000 |
| `test` | 45000 | 15000 | 15000 | 15000 | 30000 | 15000 |

## Model Configurations

| Model | Registry model | Encoding | Epochs | Batch | Elapsed | Run dir |
| --- | --- | --- | --- | --- | --- | --- |
| Stockfish-style NNUE | stockfish_nnue | simple_18 | 3 | 512 | 240.5s | results/20260428_151823_bench_signal_stockfish_style_nnue_simple18 |
| MLP simple_18 | mlp | simple_18 | 3 | 512 | 240.4s | results/20260428_152223_bench_signal_mlp_simple18 |
| CNN simple_18 | simple_cnn | simple_18 | 3 | 512 | 243.5s | results/20260428_152623_bench_signal_cnn_simple18 |
| LC0 BT4 classifier | lc0_bt4_classifier | lc0_bt4_112 | 3 | 384 | 276.1s | results/20260428_153027_bench_signal_lc0_bt4_classifier |

## Overall Test Metrics

| Model | Accuracy | F1 | Precision | Puzzle recall | PR AUC | ROC AUC | Brier | Calib err | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stockfish-style NNUE | 80.1% | 0.7355 | 65.9% | 83.2% | 0.7994 | 0.8919 | 0.1369 | 2.9% | 23548 | 6452 | 2523 | 12477 |
| MLP simple_18 | 75.1% | 0.6520 | 61.1% | 69.9% | 0.7240 | 0.8316 | 0.1698 | 3.0% | 23320 | 6680 | 4515 | 10485 |
| CNN simple_18 | 72.6% | 0.6804 | 55.7% | 87.3% | 0.7174 | 0.8478 | 0.1846 | 4.4% | 19587 | 10413 | 1898 | 13102 |
| LC0 BT4 classifier | 83.5% | 0.7742 | 71.2% | 84.9% | 0.8383 | 0.9170 | 0.1148 | 0.9% | 24838 | 5162 | 2265 | 12735 |

### Overall Interpretation

- LC0 BT4 is the strongest overall: best accuracy, F1, precision, PR AUC, ROC AUC, and calibration among this four-model run.
- NNUE is second and balanced: good verified-puzzle recall with fewer false positives than the simple CNN.
- The simple CNN has high verified-puzzle recall but pays for it with many false positives, especially on near-puzzles and tactical-looking negatives.
- The MLP is conservative on verified puzzles and misses many true puzzles compared with the others.

## Fine-Class Behavior: What They Identify

| Model | Fine class | Class accuracy | Correct target | Wrong target | Pred 0 | Pred 1 | Mean P(puzzle) | Median P(puzzle) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stockfish-style NNUE | fine 0: known non-puzzle -> target 0 | 87.8% | 13167 | 1833 | 13167 | 1833 | 17.3% | 5.8% |
| Stockfish-style NNUE | fine 1: near-puzzle -> target 0 | 69.2% | 10381 | 4619 | 10381 | 4619 | 34.5% | 26.9% |
| Stockfish-style NNUE | fine 2: verified puzzle -> target 1 | 83.2% | 12477 | 2523 | 2523 | 12477 | 74.8% | 84.2% |
| MLP simple_18 | fine 0: known non-puzzle -> target 0 | 86.9% | 13032 | 1968 | 13032 | 1968 | 27.1% | 22.0% |
| MLP simple_18 | fine 1: near-puzzle -> target 0 | 68.6% | 10288 | 4712 | 10288 | 4712 | 40.4% | 41.5% |
| MLP simple_18 | fine 2: verified puzzle -> target 1 | 69.9% | 10485 | 4515 | 4515 | 10485 | 67.0% | 72.1% |
| CNN simple_18 | fine 0: known non-puzzle -> target 0 | 76.9% | 11541 | 3459 | 11541 | 3459 | 36.0% | 34.5% |
| CNN simple_18 | fine 1: near-puzzle -> target 0 | 53.6% | 8046 | 6954 | 8046 | 6954 | 48.0% | 48.1% |
| CNN simple_18 | fine 2: verified puzzle -> target 1 | 87.3% | 13102 | 1898 | 1898 | 13102 | 67.6% | 68.9% |
| LC0 BT4 classifier | fine 0: known non-puzzle -> target 0 | 90.2% | 13524 | 1476 | 13524 | 1476 | 13.8% | 2.9% |
| LC0 BT4 classifier | fine 1: near-puzzle -> target 0 | 75.4% | 11314 | 3686 | 11314 | 3686 | 28.9% | 19.4% |
| LC0 BT4 classifier | fine 2: verified puzzle -> target 1 | 84.9% | 12735 | 2265 | 2265 | 12735 | 75.2% | 83.4% |

### False-Positive Composition

| Model | False positives | fine0 -> puzzle | fine1 -> puzzle | FP from near-puzzles | fine2 missed | Behavior |
| --- | --- | --- | --- | --- | --- | --- |
| Stockfish-style NNUE | 6452 | 1833 | 4619 | 71.6% | 2523 | balanced second-best |
| MLP simple_18 | 6680 | 1968 | 4712 | 70.5% | 4515 | conservative on true puzzles |
| CNN simple_18 | 10413 | 3459 | 6954 | 66.8% | 1898 | puzzle-sensitive, FP-heavy |
| LC0 BT4 classifier | 5162 | 1476 | 3686 | 71.4% | 2265 | best balance |

Near-puzzles are the main source of false positives for every model. This is expected from the label contract: fine `1` is deliberately target `0`, but it often resembles tactical puzzle material. The useful model is the one that can reject fine `1` without suppressing fine `2`; LC0 BT4 does that best in this run.

## Difficulty Performance

| Model | Slice | Rows | Accuracy | Puzzle recall | False pos rate | fine0 acc | fine1 acc | fine2 acc | Mean P(puzzle) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stockfish-style NNUE | hard | 9053 | 66.8% | 78.5% | 41.5% | 65.9% | 54.9% | 78.5% | 53.6% |
| Stockfish-style NNUE | very_hard | 12151 | 76.4% | 94.0% | 74.4% | 25.3% | 25.7% | 94.0% | 79.6% |
| Stockfish-style NNUE | medium | 11516 | 80.4% | 54.5% | 15.2% | 88.2% | 81.5% | 54.5% | 27.1% |
| Stockfish-style NNUE | easy | 6448 | 90.9% | 33.7% | 5.5% | 96.5% | 91.4% | 33.7% | 14.4% |
| Stockfish-style NNUE | very_easy | 5832 | 95.6% | 17.1% | 1.5% | 99.2% | 96.7% | 17.1% | 7.2% |
| MLP simple_18 | hard | 9053 | 60.8% | 68.7% | 44.8% | 62.9% | 51.4% | 68.7% | 54.3% |
| MLP simple_18 | very_hard | 12151 | 65.6% | 78.6% | 71.8% | 32.4% | 27.0% | 78.6% | 72.3% |
| MLP simple_18 | medium | 11516 | 77.7% | 45.3% | 16.8% | 86.0% | 80.4% | 45.3% | 34.7% |
| MLP simple_18 | easy | 6448 | 90.4% | 16.7% | 5.0% | 96.3% | 93.1% | 16.7% | 23.9% |
| MLP simple_18 | very_easy | 5832 | 95.1% | 5.7% | 1.5% | 98.7% | 97.9% | 5.7% | 16.1% |
| CNN simple_18 | hard | 9053 | 57.4% | 84.1% | 61.3% | 47.0% | 34.7% | 84.1% | 57.6% |
| CNN simple_18 | medium | 11516 | 67.7% | 71.0% | 32.9% | 72.4% | 62.1% | 71.0% | 45.4% |
| CNN simple_18 | very_hard | 12151 | 74.4% | 94.7% | 84.1% | 14.5% | 16.3% | 94.7% | 70.8% |
| CNN simple_18 | easy | 6448 | 81.6% | 50.4% | 16.4% | 86.4% | 79.3% | 50.4% | 36.8% |
| CNN simple_18 | very_easy | 5832 | 92.4% | 24.2% | 5.0% | 96.9% | 89.8% | 24.2% | 22.9% |
| LC0 BT4 classifier | hard | 9053 | 71.3% | 80.4% | 35.0% | 69.8% | 62.7% | 80.4% | 50.6% |
| LC0 BT4 classifier | very_hard | 12151 | 78.5% | 94.6% | 67.9% | 25.7% | 33.9% | 94.6% | 78.2% |
| LC0 BT4 classifier | medium | 11516 | 85.7% | 59.1% | 9.9% | 92.4% | 87.9% | 59.1% | 22.7% |
| LC0 BT4 classifier | easy | 6448 | 94.2% | 43.0% | 2.6% | 98.3% | 96.1% | 43.0% | 10.8% |
| LC0 BT4 classifier | very_easy | 5832 | 96.7% | 29.4% | 0.8% | 99.6% | 98.0% | 29.4% | 4.8% |

## Phase Performance

| Model | Slice | Rows | Accuracy | Puzzle recall | False pos rate | fine0 acc | fine1 acc | fine2 acc | Mean P(puzzle) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stockfish-style NNUE | endgame | 9032 | 74.5% | 87.9% | 34.7% | 77.9% | 49.0% | 87.9% | 53.1% |
| Stockfish-style NNUE | middlegame | 24918 | 81.3% | 82.1% | 19.1% | 90.8% | 70.7% | 82.1% | 40.1% |
| Stockfish-style NNUE | opening | 11050 | 81.8% | 80.5% | 17.6% | 88.9% | 77.3% | 80.5% | 38.1% |
| MLP simple_18 | endgame | 9032 | 69.6% | 88.4% | 43.3% | 70.0% | 39.4% | 88.4% | 54.7% |
| MLP simple_18 | opening | 11050 | 72.7% | 36.9% | 11.6% | 93.7% | 84.2% | 36.9% | 42.9% |
| MLP simple_18 | middlegame | 24918 | 78.2% | 75.3% | 20.5% | 90.1% | 68.7% | 75.3% | 42.1% |
| CNN simple_18 | endgame | 9032 | 64.9% | 91.7% | 53.6% | 61.1% | 27.2% | 91.7% | 57.0% |
| CNN simple_18 | opening | 11050 | 71.9% | 87.1% | 34.7% | 75.4% | 57.2% | 87.1% | 50.9% |
| CNN simple_18 | middlegame | 24918 | 75.8% | 85.4% | 28.7% | 83.1% | 59.2% | 85.4% | 48.1% |
| LC0 BT4 classifier | endgame | 9032 | 78.7% | 85.8% | 26.2% | 85.4% | 58.7% | 85.8% | 47.0% |
| LC0 BT4 classifier | opening | 11050 | 83.0% | 88.3% | 19.3% | 86.8% | 75.9% | 88.3% | 40.6% |
| LC0 BT4 classifier | middlegame | 24918 | 85.4% | 83.0% | 13.4% | 93.2% | 79.8% | 83.0% | 36.0% |

## Evaluation Bucket Performance

| Model | Slice | Rows | Accuracy | Puzzle recall | False pos rate | fine0 acc | fine1 acc | fine2 acc | Mean P(puzzle) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stockfish-style NNUE | equal | 7376 | 65.8% | 80.5% | 48.2% | 58.3% | 49.2% | 80.5% | 58.6% |
| Stockfish-style NNUE | slight_white | 7085 | 73.1% | 82.1% | 33.0% | 75.3% | 61.9% | 82.1% | 50.8% |
| Stockfish-style NNUE | slight_black | 7378 | 74.5% | 81.8% | 29.9% | 77.5% | 65.2% | 81.8% | 49.0% |
| Stockfish-style NNUE | clear_white | 9418 | 86.6% | 86.2% | 13.3% | 92.3% | 79.7% | 86.2% | 35.5% |
| Stockfish-style NNUE | clear_black | 9774 | 87.4% | 85.4% | 11.9% | 93.4% | 81.3% | 85.4% | 34.4% |
| Stockfish-style NNUE | winning_black | 1178 | 93.9% | 86.6% | 4.3% | 98.0% | 89.6% | 86.6% | 23.7% |
| Stockfish-style NNUE | winning_white | 1136 | 95.7% | 91.2% | 3.0% | 99.1% | 91.0% | 91.2% | 25.4% |
| Stockfish-style NNUE | crushing_white | 765 | 97.3% | 91.2% | 2.0% | 99.3% | 89.2% | 91.2% | 13.1% |
| Stockfish-style NNUE | crushing_black | 890 | 97.3% | 81.6% | 1.2% | 99.6% | 93.2% | 81.6% | 10.3% |
| MLP simple_18 | equal | 7376 | 59.1% | 68.8% | 50.2% | 56.6% | 47.2% | 68.8% | 57.4% |
| MLP simple_18 | slight_white | 7085 | 64.0% | 67.7% | 38.5% | 68.9% | 57.0% | 67.7% | 54.2% |
| MLP simple_18 | slight_black | 7378 | 70.9% | 68.1% | 27.4% | 80.9% | 67.1% | 68.1% | 48.3% |
| MLP simple_18 | clear_white | 9418 | 80.8% | 71.7% | 15.8% | 89.1% | 77.9% | 71.7% | 43.5% |
| MLP simple_18 | clear_black | 9774 | 85.8% | 73.5% | 9.8% | 94.9% | 84.0% | 73.5% | 35.5% |
| MLP simple_18 | winning_white | 1136 | 90.5% | 73.1% | 4.3% | 97.9% | 89.2% | 73.1% | 32.1% |
| MLP simple_18 | winning_black | 1178 | 91.9% | 71.6% | 3.2% | 97.8% | 94.2% | 71.6% | 25.8% |
| MLP simple_18 | crushing_white | 765 | 94.0% | 72.5% | 3.5% | 98.0% | 87.1% | 72.5% | 21.6% |
| MLP simple_18 | crushing_black | 890 | 96.4% | 71.1% | 1.2% | 99.7% | 92.2% | 71.1% | 15.7% |
| CNN simple_18 | equal | 7376 | 58.7% | 84.9% | 66.3% | 41.0% | 30.9% | 84.9% | 59.5% |
| CNN simple_18 | slight_white | 7085 | 64.4% | 87.8% | 51.2% | 57.0% | 43.7% | 87.8% | 56.7% |
| CNN simple_18 | slight_black | 7378 | 65.2% | 87.6% | 48.4% | 61.0% | 45.4% | 87.6% | 55.4% |
| CNN simple_18 | clear_white | 9418 | 79.2% | 89.1% | 24.5% | 81.6% | 68.0% | 89.1% | 47.9% |
| CNN simple_18 | clear_black | 9774 | 79.3% | 87.9% | 23.7% | 83.1% | 67.6% | 87.9% | 47.1% |
| CNN simple_18 | winning_black | 1178 | 92.5% | 87.1% | 6.1% | 96.6% | 86.5% | 87.1% | 34.1% |
| CNN simple_18 | winning_white | 1136 | 94.2% | 90.8% | 4.8% | 97.7% | 87.9% | 90.8% | 35.2% |
| CNN simple_18 | crushing_black | 890 | 96.9% | 82.9% | 1.8% | 99.6% | 88.3% | 82.9% | 19.1% |
| CNN simple_18 | crushing_white | 765 | 97.0% | 90.0% | 2.2% | 99.0% | 90.3% | 90.0% | 21.4% |
| LC0 BT4 classifier | equal | 7376 | 70.4% | 82.8% | 41.5% | 59.8% | 58.0% | 82.8% | 56.6% |
| LC0 BT4 classifier | slight_black | 7378 | 78.1% | 83.2% | 25.0% | 80.5% | 71.3% | 83.2% | 46.0% |
| LC0 BT4 classifier | slight_white | 7085 | 78.2% | 84.8% | 26.2% | 80.0% | 70.0% | 84.8% | 48.1% |
| LC0 BT4 classifier | clear_white | 9418 | 89.8% | 87.4% | 9.3% | 95.4% | 84.8% | 87.4% | 32.0% |
| LC0 BT4 classifier | clear_black | 9774 | 90.1% | 86.0% | 8.5% | 95.7% | 86.1% | 86.0% | 30.7% |
| LC0 BT4 classifier | winning_black | 1178 | 95.2% | 89.2% | 3.3% | 98.4% | 92.3% | 89.2% | 21.2% |
| LC0 BT4 classifier | winning_white | 1136 | 95.9% | 91.9% | 3.0% | 98.9% | 91.5% | 91.9% | 23.9% |
| LC0 BT4 classifier | crushing_white | 765 | 96.9% | 91.2% | 2.5% | 99.0% | 88.2% | 91.2% | 12.3% |
| LC0 BT4 classifier | crushing_black | 890 | 97.6% | 84.2% | 1.1% | 99.7% | 93.2% | 84.2% | 9.0% |

## Tactical Motif Performance

Motif rows are multi-label: a position with `fork|hanging` contributes to both motif rows.
### Stockfish-style NNUE

| Type | Motif | Rows | Accuracy | Puzzle recall | False pos rate | Near-puzzle acc | Puzzle acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| worst | mate_in_1 | 2077 | 72.6% | 91.7% | 37.1% | 53.7% | 91.7% |
| worst | promotion | 1211 | 73.8% | 73.4% | 26.0% | 66.2% | 73.4% |
| worst | underpromotion | 1211 | 73.8% | 73.4% | 26.0% | 66.2% | 73.4% |
| worst | overload | 3790 | 77.0% | 78.8% | 23.8% | 70.3% | 78.8% |
| worst | pin | 10830 | 78.2% | 82.4% | 23.7% | 69.4% | 82.4% |
| worst | discovered_attack | 3439 | 78.4% | 82.4% | 23.3% | 71.7% | 82.4% |
| worst | skewer | 9054 | 78.9% | 82.7% | 22.9% | 70.8% | 82.7% |
| worst | (none) | 8957 | 79.4% | 80.3% | 20.8% | 64.4% | 80.3% |
| best | fork | 15648 | 80.5% | 85.1% | 21.5% | 70.9% | 85.1% |
| best | hanging | 24190 | 80.3% | 85.1% | 22.9% | 69.0% | 85.1% |
| best | (none) | 8957 | 79.4% | 80.3% | 20.8% | 64.4% | 80.3% |
| best | skewer | 9054 | 78.9% | 82.7% | 22.9% | 70.8% | 82.7% |
| best | discovered_attack | 3439 | 78.4% | 82.4% | 23.3% | 71.7% | 82.4% |
| best | pin | 10830 | 78.2% | 82.4% | 23.7% | 69.4% | 82.4% |
| best | overload | 3790 | 77.0% | 78.8% | 23.8% | 70.3% | 78.8% |
| best | promotion | 1211 | 73.8% | 73.4% | 26.0% | 66.2% | 73.4% |

### MLP simple_18

| Type | Motif | Rows | Accuracy | Puzzle recall | False pos rate | Near-puzzle acc | Puzzle acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| worst | promotion | 1211 | 62.6% | 76.3% | 41.5% | 50.0% | 76.3% |
| worst | underpromotion | 1211 | 62.6% | 76.3% | 41.5% | 50.0% | 76.3% |
| worst | mate_in_1 | 2077 | 65.8% | 84.1% | 43.4% | 45.8% | 84.1% |
| worst | discovered_attack | 3439 | 73.0% | 67.0% | 24.4% | 69.2% | 67.0% |
| worst | overload | 3790 | 73.3% | 68.4% | 24.4% | 69.3% | 68.4% |
| worst | skewer | 9054 | 73.5% | 68.7% | 24.1% | 68.8% | 68.7% |
| worst | pin | 10830 | 73.6% | 68.9% | 24.1% | 69.0% | 68.9% |
| worst | hanging | 24190 | 74.5% | 69.8% | 22.4% | 69.5% | 69.8% |
| best | fork | 15648 | 76.6% | 72.4% | 21.6% | 70.3% | 72.4% |
| best | (none) | 8957 | 75.6% | 72.8% | 23.4% | 61.6% | 72.8% |
| best | hanging | 24190 | 74.5% | 69.8% | 22.4% | 69.5% | 69.8% |
| best | pin | 10830 | 73.6% | 68.9% | 24.1% | 69.0% | 68.9% |
| best | skewer | 9054 | 73.5% | 68.7% | 24.1% | 68.8% | 68.7% |
| best | overload | 3790 | 73.3% | 68.4% | 24.4% | 69.3% | 68.4% |
| best | discovered_attack | 3439 | 73.0% | 67.0% | 24.4% | 69.2% | 67.0% |
| best | mate_in_1 | 2077 | 65.8% | 84.1% | 43.4% | 45.8% | 84.1% |

### CNN simple_18

| Type | Motif | Rows | Accuracy | Puzzle recall | False pos rate | Near-puzzle acc | Puzzle acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| worst | promotion | 1211 | 38.4% | 97.1% | 79.1% | 13.2% | 97.1% |
| worst | underpromotion | 1211 | 38.4% | 97.1% | 79.1% | 13.2% | 97.1% |
| worst | mate_in_1 | 2077 | 64.5% | 93.8% | 50.3% | 40.7% | 93.8% |
| worst | discovered_attack | 3439 | 69.0% | 85.0% | 37.9% | 53.8% | 85.0% |
| worst | overload | 3790 | 69.4% | 85.4% | 38.2% | 54.4% | 85.4% |
| worst | skewer | 9054 | 70.2% | 87.9% | 38.3% | 53.3% | 87.9% |
| worst | pin | 10830 | 70.6% | 86.9% | 37.1% | 54.3% | 86.9% |
| worst | (none) | 8957 | 72.2% | 84.3% | 32.0% | 49.9% | 84.3% |
| best | hanging | 24190 | 74.0% | 88.8% | 35.7% | 54.2% | 88.8% |
| best | fork | 15648 | 72.9% | 89.2% | 34.2% | 56.2% | 89.2% |
| best | (none) | 8957 | 72.2% | 84.3% | 32.0% | 49.9% | 84.3% |
| best | pin | 10830 | 70.6% | 86.9% | 37.1% | 54.3% | 86.9% |
| best | skewer | 9054 | 70.2% | 87.9% | 38.3% | 53.3% | 87.9% |
| best | overload | 3790 | 69.4% | 85.4% | 38.2% | 54.4% | 85.4% |
| best | discovered_attack | 3439 | 69.0% | 85.0% | 37.9% | 53.8% | 85.0% |
| best | mate_in_1 | 2077 | 64.5% | 93.8% | 50.3% | 40.7% | 93.8% |

### LC0 BT4 classifier

| Type | Motif | Rows | Accuracy | Puzzle recall | False pos rate | Near-puzzle acc | Puzzle acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| worst | promotion | 1211 | 75.3% | 70.9% | 23.4% | 69.0% | 70.9% |
| worst | underpromotion | 1211 | 75.3% | 70.9% | 23.4% | 69.0% | 70.9% |
| worst | mate_in_1 | 2077 | 77.4% | 92.1% | 30.1% | 62.7% | 92.1% |
| worst | skewer | 9054 | 82.1% | 83.4% | 18.6% | 76.1% | 83.4% |
| worst | pin | 10830 | 82.1% | 84.9% | 19.3% | 75.3% | 84.9% |
| worst | overload | 3790 | 82.1% | 80.7% | 17.2% | 78.7% | 80.7% |
| worst | (none) | 8957 | 82.3% | 79.6% | 16.8% | 70.8% | 79.6% |
| worst | discovered_attack | 3439 | 82.4% | 84.0% | 18.2% | 78.2% | 84.0% |
| best | fork | 15648 | 84.7% | 85.9% | 15.8% | 78.2% | 85.9% |
| best | hanging | 24190 | 84.1% | 87.5% | 18.1% | 75.8% | 87.5% |
| best | discovered_attack | 3439 | 82.4% | 84.0% | 18.2% | 78.2% | 84.0% |
| best | (none) | 8957 | 82.3% | 79.6% | 16.8% | 70.8% | 79.6% |
| best | overload | 3790 | 82.1% | 80.7% | 17.2% | 78.7% | 80.7% |
| best | pin | 10830 | 82.1% | 84.9% | 19.3% | 75.3% | 84.9% |
| best | skewer | 9054 | 82.1% | 83.4% | 18.6% | 76.1% | 83.4% |
| best | mate_in_1 | 2077 | 77.4% | 92.1% | 30.1% | 62.7% | 92.1% |

## Best/Worst Slice Summary

| Model | Weakest slices | Strongest slices |
| --- | --- | --- |
| Stockfish-style NNUE | crtk_eval_bucket=equal 65.8% (7376 rows); crtk_difficulty=hard 66.8% (9053 rows); crtk_tactic_motifs=mate_in_1 72.6% (2077 rows); crtk_eval_bucket=slight_white 73.1% (7085 rows); crtk_tactic_motifs=promotion 73.8% (1211 rows) | crtk_eval_bucket=crushing_black 97.3% (890 rows); crtk_eval_bucket=crushing_white 97.2% (765 rows); crtk_eval_bucket=winning_white 95.7% (1136 rows); crtk_difficulty=very_easy 95.6% (5832 rows); crtk_eval_bucket=winning_black 93.9% (1178 rows) |
| MLP simple_18 | crtk_eval_bucket=equal 59.1% (7376 rows); crtk_difficulty=hard 60.8% (9053 rows); crtk_tactic_motifs=promotion 62.6% (1211 rows); crtk_tactic_motifs=underpromotion 62.6% (1211 rows); crtk_tag_families=THREAT 62.6% (1211 rows) | crtk_eval_bucket=crushing_black 96.4% (890 rows); crtk_difficulty=very_easy 95.1% (5832 rows); crtk_eval_bucket=crushing_white 94.0% (765 rows); crtk_eval_bucket=winning_black 91.8% (1178 rows); crtk_eval_bucket=winning_white 90.5% (1136 rows) |
| CNN simple_18 | crtk_tactic_motifs=promotion 38.4% (1211 rows); crtk_tactic_motifs=underpromotion 38.4% (1211 rows); crtk_tag_families=THREAT 38.4% (1211 rows); crtk_difficulty=hard 57.4% (9053 rows); crtk_eval_bucket=equal 58.7% (7376 rows) | crtk_eval_bucket=crushing_white 97.0% (765 rows); crtk_eval_bucket=crushing_black 96.9% (890 rows); crtk_eval_bucket=winning_white 94.2% (1136 rows); crtk_eval_bucket=winning_black 92.5% (1178 rows); crtk_difficulty=very_easy 92.4% (5832 rows) |
| LC0 BT4 classifier | crtk_eval_bucket=equal 70.4% (7376 rows); crtk_difficulty=hard 71.3% (9053 rows); crtk_tactic_motifs=promotion 75.3% (1211 rows); crtk_tactic_motifs=underpromotion 75.3% (1211 rows); crtk_tag_families=THREAT 75.3% (1211 rows) | crtk_eval_bucket=crushing_black 97.6% (890 rows); crtk_eval_bucket=crushing_white 96.9% (765 rows); crtk_difficulty=very_easy 96.7% (5832 rows); crtk_eval_bucket=winning_white 95.9% (1136 rows); crtk_eval_bucket=winning_black 95.2% (1178 rows) |

## Agreement And Consensus Errors

| Models correct on row | Rows |
| ---: | ---: |
| 0 | 2954 |
| 1 | 3500 |
| 2 | 4462 |
| 3 | 8668 |
| 4 | 25416 |

- Rows all four models missed: `2954`
- All-wrong by fine class: `{0: 603, 1: 1783, 2: 568}`
- All-wrong by difficulty: `{'very_hard': 1377, 'hard': 875, 'medium': 465, 'easy': 127, 'very_easy': 110}`
- All-wrong by eval bucket: `{'equal': 772, 'slight_black': 558, 'slight_white': 551, 'clear_black': 485, 'clear_white': 474, 'winning_black': 41, 'winning_white': 38, 'crushing_black': 19}`

### Highest Average Puzzle Probability Among Consensus Errors

| Sample | Fine | Target | Avg P(puzzle) | Difficulty | Phase | Eval | Motifs | FEN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stack-353000.json:1951 | 1 | 0 | 98.2% | very_hard | middlegame | crushing_white | fork\|hanging\|promotion\|underpromotion | 2r1r1k1/1p1bpp1p/6p1/p3B3/8/P4B1P/1PQp1PP1/R3R1K1 b - - 0 24 |
| Stack-975000.json:16352 | 0 | 0 | 97.2% | very_hard | endgame | crushing_white | hanging | 8/p5pp/2p2p2/P2p1k2/3P1Q2/2P4P/1P4PK/8 b - - 0 34 |
| Stack-1122000.json:15371 | 1 | 0 | 97.1% | very_hard | middlegame | crushing_black | hanging | 4r1k1/1br2ppp/pN6/npp5/4p3/1PP1q1NP/P1B5/3RR1K1 w - - 0 26 |
| Stack-586000.json:8728 | 0 | 0 | 96.7% | very_hard | middlegame | winning_black | hanging | 1k1r4/5pp1/1p3n2/p1ppq3/P1P4P/1P1P4/5PK1/4RR2 w - - 0 26 |
| Stack-430000.json:17879 | 1 | 0 | 96.6% | very_hard | middlegame | crushing_white | hanging | r1Q3k1/1p1n1pPp/8/p6q/5P2/7P/PP2N3/2KRB2R b - - 0 23 |
| Stack-1546000.json:18287 | 0 | 0 | 96.5% | very_hard | endgame | clear_white | hanging | 4k3/pp2R3/2p3p1/5p2/8/1P5P/P5P1/7K b - - 0 41 |
| Stack-1126000.json:14528 | 0 | 0 | 96.4% | very_hard | middlegame | crushing_white | hanging\|pin | r4n2/pp1b2k1/2p1p1p1/3pPQ2/3P4/2P5/PP3P1P/1K1R2R1 b - - 0 27 |
| Stack-914000.json:15673 | 0 | 0 | 96.4% | very_hard | endgame | clear_white | hanging | 3R2k1/4bpp1/1p5p/p3Pb2/2P2B2/5N1P/5PP1/6K1 b - - 0 26 |
| Stack-1157000.json:19637 | 1 | 0 | 96.2% | very_hard | endgame | crushing_white | mate_in_1\|pin\|promotion\|underpromotion | 6k1/pp4pp/8/4P1Q1/2P5/2P5/P3p1PP/7K b - - 0 28 |
| Stack-413000.json:22399 | 1 | 0 | 96.0% | very_hard | endgame | crushing_black | hanging | 8/8/3k1p2/6p1/2p3P1/2P2P1P/5K2/4q3 w - - 0 51 |
| Stack-1421000.json:20702 | 1 | 0 | 96.0% | very_hard | middlegame | clear_white | hanging | r1b2r2/p4p2/1p2pkB1/3p2Q1/8/8/P4KP1/2R4R b - - 0 26 |
| Stack-890000.json:13764 | 1 | 0 | 95.9% | very_hard | middlegame | clear_white | hanging\|pin | 3rkbR1/pp1n1p2/2p4r/4Q3/8/8/PPP3PP/5R1K b - - 0 23 |

## Highest-Confidence Wrong Rows Per Model

### Stockfish-style NNUE

| Sample | Fine | Target | Pred | Confidence | P(puzzle) | Difficulty | Phase | Eval | Motifs | FEN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stack-1177000.json:9100 | 2 | 1 | 0 | 99.7% | 0.3% | very_easy | middlegame | clear_white |  | k2r3r/pb1B2p1/1p2P3/8/4p3/6p1/PP4Q1/2R2R1K w - - 0 33 |
| Stack-1209000.json:7477 | 2 | 1 | 0 | 99.6% | 0.4% | medium | middlegame | clear_black | hanging | 5rk1/pp4p1/2pr1nbp/2P1R1p1/3P4/PB5P/5PP1/3R2K1 b - - 0 25 |
| Stack-338000.json:4549 | 2 | 1 | 0 | 99.6% | 0.4% | easy | opening | clear_white | hanging | r2qk1r1/pp4p1/5p2/2N1pn1p/4P3/P3K1P1/1P5P/1RBQ2NR w - - 1 19 |
| Stack-1571000.json:1 | 2 | 1 | 0 | 99.6% | 0.4% | very_easy | middlegame | winning_black |  | r2qk2r/pp5p/2nbp3/2pn3B/8/6P1/PPPP3P/RNB2RK1 b kq - 0 16 |
| Stack-821000.json:13178 | 2 | 1 | 0 | 99.6% | 0.4% | hard | middlegame | equal | hanging\|pin\|skewer | 5r2/1p3k2/r4pp1/p3pR2/P7/5N2/1PP4P/1K4R1 w - - 1 29 |
| Stack-24000.json:19924 | 2 | 1 | 0 | 99.6% | 0.4% | very_easy | middlegame | clear_black | pin | r5k1/4b1pp/p1q2p2/4n1r1/8/1Q6/PP3PPP/2R2RK1 b - - 1 24 |
| Stack-1037000.json:21496 | 2 | 1 | 0 | 99.5% | 0.5% | medium | middlegame | clear_white | hanging | 1r5r/1pp2k2/3p3p/p3p1p1/2P3N1/P2P2P1/1Pn1PPBP/R3K2R w KQ - 1 19 |
| Stack-1112000.json:2829 | 2 | 1 | 0 | 99.5% | 0.5% | medium | opening | slight_black | fork\|hanging\|overload | r4rk1/p3qppp/3p1n2/2pNn3/P3P3/3b4/1PP2PPP/R1BQ1R1K b - - 1 15 |
| Stack-1107000.json:1849 | 2 | 1 | 0 | 99.5% | 0.5% | very_easy | middlegame | crushing_black |  | 5r2/5bk1/q1pR4/6R1/4p3/4P2P/2r2PPK/8 b - - 0 35 |
| Stack-1569000.json:14166 | 2 | 1 | 0 | 99.3% | 0.7% | easy | endgame | clear_black | hanging | N7/8/1p2k3/3b4/3p4/8/5K2/8 b - - 0 64 |
| Stack-1466000.json:4770 | 2 | 1 | 0 | 99.3% | 0.7% | very_easy | middlegame | crushing_black |  | 3r1b1r/kb3ppp/1pp2n2/3p4/Qp6/3BP3/P2BKPPP/7q b - - 1 19 |
| Stack-1326000.json:1567 | 2 | 1 | 0 | 99.3% | 0.7% | very_easy | middlegame | clear_white | pin | 2k4r/1pp2Q2/p2b4/3P4/2P5/5pR1/PP3P1K/8 w - - 2 34 |

### MLP simple_18

| Sample | Fine | Target | Pred | Confidence | P(puzzle) | Difficulty | Phase | Eval | Motifs | FEN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stack-1107000.json:1849 | 2 | 1 | 0 | 99.3% | 0.7% | very_easy | middlegame | crushing_black |  | 5r2/5bk1/q1pR4/6R1/4p3/4P2P/2r2PPK/8 b - - 0 35 |
| Stack-1461000.json:5939 | 2 | 1 | 0 | 99.1% | 0.9% | very_easy | middlegame | winning_black |  | 2kr3r/pBp2ppp/4pb2/8/2K3P1/2P5/1R3P1P/8 b - - 0 29 |
| Stack-31000.json:12481 | 2 | 1 | 0 | 98.9% | 1.1% | very_easy | middlegame | winning_black | pin | r1b3k1/1p3p1P/p3p3/3p4/1q3Pn1/b2B4/1PP3P1/1K1N2NR b - - 0 20 |
| Stack-24000.json:19924 | 2 | 1 | 0 | 98.8% | 1.2% | very_easy | middlegame | clear_black | pin | r5k1/4b1pp/p1q2p2/4n1r1/8/1Q6/PP3PPP/2R2RK1 b - - 1 24 |
| Stack-1623000.json:4412 | 2 | 1 | 0 | 98.7% | 1.3% | medium | middlegame | clear_black |  | r1b1q2k/5Npp/p1n5/1pp1p3/8/2PP2P1/PP4PP/RN3R1K b - - 1 20 |
| Stack-1546000.json:18287 | 0 | 0 | 1 | 98.6% | 98.6% | very_hard | endgame | clear_white | hanging | 4k3/pp2R3/2p3p1/5p2/8/1P5P/P5P1/7K b - - 0 41 |
| Stack-365000.json:12522 | 2 | 1 | 0 | 98.6% | 1.4% | medium | opening | clear_black | hanging | r1bq3r/ppp1bkpp/8/4p3/4n3/5Q1P/PPP2PP1/RNB2RK1 b - - 1 10 |
| Stack-1497000.json:12404 | 2 | 1 | 0 | 98.6% | 1.4% | easy | middlegame | clear_black | hanging\|pin | rn2r1k1/1pp2pnP/p2p4/8/PP1P4/2P4b/2B3PP/RN3K2 b - - 0 21 |
| Stack-914000.json:15673 | 0 | 0 | 1 | 98.6% | 98.6% | very_hard | endgame | clear_white | hanging | 3R2k1/4bpp1/1p5p/p3Pb2/2P2B2/5N1P/5PP1/6K1 b - - 0 26 |
| Stack-1578000.json:6565 | 2 | 1 | 0 | 98.5% | 1.5% | easy | middlegame | clear_black |  | r7/p1r2ppp/1p2k3/3np3/1PPp3P/1P3BP1/3B1P2/bN3RK1 b - c3 0 27 |
| Stack-1297000.json:9644 | 2 | 1 | 0 | 98.5% | 1.5% | very_easy | endgame | winning_black | hanging | 3k4/ppp2p1p/8/8/2qR4/6P1/5P2/6K1 b - - 0 27 |
| Stack-1253000.json:23046 | 0 | 0 | 1 | 98.5% | 98.5% | very_hard | middlegame | clear_white | hanging | r5k1/1p3ppp/p2b1n2/2pP4/P4Q2/3B4/1P3PPP/R5K1 b - - 0 20 |

### CNN simple_18

| Sample | Fine | Target | Pred | Confidence | P(puzzle) | Difficulty | Phase | Eval | Motifs | FEN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stack-1209000.json:13968 | 0 | 0 | 1 | 98.1% | 98.1% | very_hard | middlegame | winning_white | hanging\|overload\|pin | r3k2r/pb1pQpb1/1p2p1pp/1B6/8/2P5/PP3PPP/RN3RK1 b kq - 0 16 |
| Stack-353000.json:1951 | 1 | 0 | 1 | 96.9% | 96.9% | very_hard | middlegame | crushing_white | fork\|hanging\|promotion\|underpromotion | 2r1r1k1/1p1bpp1p/6p1/p3B3/8/P4B1P/1PQp1PP1/R3R1K1 b - - 0 24 |
| Stack-1299000.json:24320 | 0 | 0 | 1 | 96.9% | 96.9% | very_hard | endgame | clear_white | hanging\|pin | 7r/p7/2k1pR2/3Q4/2pP4/2P1p3/P6p/7K b - - 0 34 |
| Stack-1441000.json:5622 | 1 | 0 | 1 | 96.8% | 96.8% | very_hard | endgame | crushing_white | hanging | 3Q4/1R2k3/8/8/8/7P/rpp1N1PK/8 b - - 0 54 |
| Stack-1171000.json:22286 | 1 | 0 | 1 | 96.8% | 96.8% | very_hard | middlegame | clear_black | hanging | 7r/1bkp3p/1p2p3/4P3/3P1B2/8/PPP2q2/R3KB2 w Q - 0 29 |
| Stack-1122000.json:15371 | 1 | 0 | 1 | 96.5% | 96.5% | very_hard | middlegame | crushing_black | hanging | 4r1k1/1br2ppp/pN6/npp5/4p3/1PP1q1NP/P1B5/3RR1K1 w - - 0 26 |
| Stack-430000.json:17879 | 1 | 0 | 1 | 95.6% | 95.6% | very_hard | middlegame | crushing_white | hanging | r1Q3k1/1p1n1pPp/8/p6q/5P2/7P/PP2N3/2KRB2R b - - 0 23 |
| Stack-170000.json:17673 | 1 | 0 | 1 | 95.0% | 95.0% | very_hard | endgame | clear_black | hanging | 8/8/4p2P/3kP3/8/8/4K2P/5q2 w - - 0 57 |
| Stack-413000.json:22399 | 1 | 0 | 1 | 94.6% | 94.6% | very_hard | endgame | crushing_black | hanging | 8/8/3k1p2/6p1/2p3P1/2P2P1P/5K2/4q3 w - - 0 51 |
| Stack-729000.json:8792 | 1 | 0 | 1 | 94.6% | 94.6% | very_hard | middlegame | crushing_white | hanging | k3r3/pQ6/5p2/5P1p/3B4/1PP4P/P3p1n1/1K2R3 b - - 0 46 |
| Stack-1307000.json:17463 | 1 | 0 | 1 | 94.5% | 94.5% | very_hard | endgame | clear_black | hanging | 8/8/8/3k4/8/7P/5KP1/4q3 w - - 0 67 |
| Stack-214000.json:7762 | 0 | 0 | 1 | 94.3% | 94.3% | very_hard | opening | slight_white | hanging | rnbqkbnr/pp3Bp1/2pp3p/8/3pP3/8/PPPN1PPP/R1BQK1NR b KQkq - 0 6 |

### LC0 BT4 classifier

| Sample | Fine | Target | Pred | Confidence | P(puzzle) | Difficulty | Phase | Eval | Motifs | FEN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stack-24000.json:19924 | 2 | 1 | 0 | 99.8% | 0.2% | very_easy | middlegame | clear_black | pin | r5k1/4b1pp/p1q2p2/4n1r1/8/1Q6/PP3PPP/2R2RK1 b - - 1 24 |
| Stack-1466000.json:4770 | 2 | 1 | 0 | 99.8% | 0.2% | very_easy | middlegame | crushing_black |  | 3r1b1r/kb3ppp/1pp2n2/3p4/Qp6/3BP3/P2BKPPP/7q b - - 1 19 |
| Stack-1571000.json:1 | 2 | 1 | 0 | 99.8% | 0.2% | very_easy | middlegame | winning_black |  | r2qk2r/pp5p/2nbp3/2pn3B/8/6P1/PPPP3P/RNB2RK1 b kq - 0 16 |
| Stack-688000.json:21865 | 2 | 1 | 0 | 99.6% | 0.4% | medium | middlegame | clear_black | hanging\|overload | 6k1/rQ2n3/pp2pq1p/4pPp1/P1P5/8/5PPP/3R2K1 b - - 1 25 |
| Stack-1531000.json:2461 | 2 | 1 | 0 | 99.6% | 0.4% | very_easy | endgame | crushing_black |  | 8/pp3p2/2p3pk/2np4/3K1P2/8/P1r5/7R b - - 1 41 |
| Stack-1571000.json:10000 | 1 | 0 | 1 | 99.5% | 99.5% | very_hard | middlegame | winning_black | fork\|hanging | r4rk1/ppp2ppp/3p4/3q1N2/8/2P5/P1P2PPP/R4RK1 w - - 0 16 |
| Stack-685000.json:1467 | 2 | 1 | 0 | 99.5% | 0.5% | medium | middlegame | clear_black | hanging | 5r2/p2qp1bk/1p1pQ1pp/1Pp1p3/P1P1P3/2P2r1P/5PP1/R4RK1 b - - 1 24 |
| Stack-353000.json:1951 | 1 | 0 | 1 | 99.4% | 99.4% | very_hard | middlegame | crushing_white | fork\|hanging\|promotion\|underpromotion | 2r1r1k1/1p1bpp1p/6p1/p3B3/8/P4B1P/1PQp1PP1/R3R1K1 b - - 0 24 |
| Stack-154000.json:17821 | 2 | 1 | 0 | 99.4% | 0.6% | very_easy | middlegame | crushing_white | hanging\|overload | 2rk4/1Q3ppp/p2r4/3b1PP1/8/1P6/P1PP3P/R3R2K w - - 1 27 |
| Stack-821000.json:13178 | 2 | 1 | 0 | 99.4% | 0.6% | hard | middlegame | equal | hanging\|pin\|skewer | 5r2/1p3k2/r4pp1/p3pR2/P7/5N2/1PP4P/1K4R1 w - - 1 29 |
| Stack-1355000.json:369 | 2 | 1 | 0 | 99.4% | 0.6% | very_easy | middlegame | winning_black |  | 4r3/ppR4k/7r/6R1/5pP1/3pn3/8/6K1 b - - 1 44 |
| Stack-944000.json:6224 | 2 | 1 | 0 | 99.3% | 0.7% | easy | middlegame | clear_black |  | r2q1r1k/5p2/p4b1p/3p4/1p1nQ3/3B4/PP3PPP/1R3RK1 b - - 1 24 |

## Practical Conclusions

1. **Use LC0 BT4 as the current benchmark winner.** It has the best overall metrics and the best fine-class balance: it rejects both known non-puzzles and near-puzzles better while retaining strong verified-puzzle recall.
2. **Keep NNUE as the compact/fast baseline.** It is materially behind LC0 but still much cleaner than the simple CNN on false positives.
3. **Treat the simple CNN as recall-biased.** It catches many verified puzzles, but it overpredicts puzzle on near-puzzles and promotion/underpromotion-like negatives. This makes it useful as a high-recall candidate generator, not as the best calibrated classifier.
4. **Hard negatives are the key bottleneck.** Fine `1` near-puzzles and very-hard tactical-looking non-puzzles produce most false positives. Any new idea should be judged by whether it improves fine `1` rejection without reducing fine `2` recall.
5. **Easy/very-easy verified puzzles are paradoxically under-recalled by some models.** The dataset difficulty labels are not simply “easy for the model”; many very-easy rows are confidently treated as non-puzzles, suggesting representation blind spots or source-label distribution effects.
6. **Promotion/underpromotion and mate-in-1 slices deserve targeted ablations.** They are consistently among the weakest motif slices, especially for simple CNN and MLP.

## Artifact Pointers

| Model | Run summary | Test slice report | Tagged predictions | Final metrics |
| --- | --- | --- | --- | --- |
| Stockfish-style NNUE | results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/run_summary.md | results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/slice_report_test.md | results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/predictions_test_crtk_tags.parquet | results/20260428_151823_bench_signal_stockfish_style_nnue_simple18/metrics_final.json |
| MLP simple_18 | results/20260428_152223_bench_signal_mlp_simple18/run_summary.md | results/20260428_152223_bench_signal_mlp_simple18/slice_report_test.md | results/20260428_152223_bench_signal_mlp_simple18/predictions_test_crtk_tags.parquet | results/20260428_152223_bench_signal_mlp_simple18/metrics_final.json |
| CNN simple_18 | results/20260428_152623_bench_signal_cnn_simple18/run_summary.md | results/20260428_152623_bench_signal_cnn_simple18/slice_report_test.md | results/20260428_152623_bench_signal_cnn_simple18/predictions_test_crtk_tags.parquet | results/20260428_152623_bench_signal_cnn_simple18/metrics_final.json |
| LC0 BT4 classifier | results/20260428_153027_bench_signal_lc0_bt4_classifier/run_summary.md | results/20260428_153027_bench_signal_lc0_bt4_classifier/slice_report_test.md | results/20260428_153027_bench_signal_lc0_bt4_classifier/predictions_test_crtk_tags.parquet | results/20260428_153027_bench_signal_lc0_bt4_classifier/metrics_final.json |
