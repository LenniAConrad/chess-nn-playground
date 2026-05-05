# Benchmark Data Readiness

- Status: `ready`
- Split dir: `data/splits/crtk_sample_3class_unique_crtk_tags`
- CRTK metadata is benchmark metadata only and must not be used as model input.

## Split Summary

| split | rows | fine_counts | duplicate_fens | invalid_fens |
| --- | --- | --- | --- | --- |
| train | 360000 | {"0": 120000, "1": 120000, "2": 120000} | 0 | 0 |
| val | 45000 | {"0": 15000, "1": 15000, "2": 15000} | 0 | 0 |
| test | 45000 | {"0": 15000, "1": 15000, "2": 15000} | 0 | 0 |

## Cross Split Checks

| check | value |
| --- | --- |
| sample_id_overlap | {'test__train': 0, 'test__val': 0, 'train__val': 0} |
| split_group_id_overlap | {'test__train': 0, 'test__val': 0, 'train__val': 0} |
| normalized_fen_overlap | {'test__train': 0, 'test__val': 0, 'train__val': 0} |
| label_conflicting_fens | 0 |

## Benchmark Metadata Coverage

| split | difficulty | phase |
| --- | --- | --- |
| train | {"easy": 51870, "hard": 72929, "medium": 89426, "very_easy": 47588, "very_hard": 98187} | {"endgame": 71996, "middlegame": 199018, "opening": 88986} |
| val | {"easy": 6266, "hard": 9170, "medium": 11338, "very_easy": 5943, "very_hard": 12283} | {"endgame": 9112, "middlegame": 24868, "opening": 11020} |
| test | {"easy": 6448, "hard": 9053, "medium": 11516, "very_easy": 5832, "very_hard": 12151} | {"endgame": 9032, "middlegame": 24918, "opening": 11050} |

## Issues

No blocking issues found.
