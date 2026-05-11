# Benchmark Data Readiness

- Status: `blocked`
- Split dir: `data/splits/crtk_sample_3class_unique_crtk_tags`
- CRTK metadata is benchmark metadata only and must not be used as model input.

## Split Summary

| split | rows | fine_counts | duplicate_fens | invalid_fens |
| --- | --- | --- | --- | --- |
| train | 173029 | {"0": 57971, "1": 57519, "2": 57539} | 0 | 0 |
| val | 21305 | {"0": 7189, "1": 7012, "2": 7104} | 0 | 0 |
| test | 21501 | {"0": 7206, "1": 7240, "2": 7055} | 0 | 0 |

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
| train | {"easy": 25094, "hard": 34738, "medium": 43210, "very_easy": 22990, "very_hard": 46997} | {"endgame": 34910, "middlegame": 95929, "opening": 42190} |
| val | {"easy": 3039, "hard": 4256, "medium": 5478, "very_easy": 2865, "very_hard": 5667} | {"endgame": 4243, "middlegame": 11854, "opening": 5208} |
| test | {"easy": 3190, "hard": 4181, "medium": 5482, "very_easy": 2788, "very_hard": 5860} | {"endgame": 4338, "middlegame": 11835, "opening": 5328} |

## Issues

- train: fine_label counts {1: 57519, 0: 57971, 2: 57539} do not match expected {0: 120000, 1: 120000, 2: 120000}
- val: fine_label counts {1: 7012, 2: 7104, 0: 7189} do not match expected {0: 15000, 1: 15000, 2: 15000}
- test: fine_label counts {1: 7240, 2: 7055, 0: 7206} do not match expected {0: 15000, 1: 15000, 2: 15000}
