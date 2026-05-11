# CRTK Tagged Split Report

These tags are metadata for benchmarking and error analysis only. They must not be used as neural-network input features.

## Files

- `data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet`
- `data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet`
- `data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet`

## train

| field | value |
| --- | --- |
| rows | 173029 |
| tagged_rows | 173029 |
| rows_with_tactic_motif | 138193 |

### Difficulty

| value | count |
| --- | --- |
| very_hard | 46997 |
| medium | 43210 |
| hard | 34738 |
| easy | 25094 |
| very_easy | 22990 |

### Phase

| value | count |
| --- | --- |
| middlegame | 95929 |
| opening | 42190 |
| endgame | 34910 |

### Eval Bucket

| value | count |
| --- | --- |
| clear_black | 37805 |
| clear_white | 36528 |
| equal | 28128 |
| slight_black | 27952 |
| slight_white | 27103 |
| winning_black | 4618 |
| winning_white | 4509 |
| crushing_black | 3232 |
| crushing_white | 3154 |

### Tactical Motifs

| value | count |
| --- | --- |
| hanging | 92407 |
| fork | 60072 |
| pin | 41946 |
| skewer | 34998 |
| (none) | 34836 |
| overload | 14393 |
| discovered_attack | 13269 |
| mate_in_1 | 8235 |
| promotion | 4887 |
| underpromotion | 4887 |

### Tag Families

| value | count |
| --- | --- |
| DEVELOPMENT | 173029 |
| FACT | 173029 |
| INITIATIVE | 173029 |
| KING | 173029 |
| MATERIAL | 173029 |
| META | 173029 |
| MOBILITY | 173029 |
| PAWN | 173029 |
| PIECE | 173029 |
| SPACE | 173029 |
| TACTIC | 138193 |
| ENDGAME | 54817 |
| OUTPOST | 31060 |
| THREAT | 4887 |

## val

| field | value |
| --- | --- |
| rows | 21305 |
| tagged_rows | 21305 |
| rows_with_tactic_motif | 16997 |

### Difficulty

| value | count |
| --- | --- |
| very_hard | 5667 |
| medium | 5478 |
| hard | 4256 |
| easy | 3039 |
| very_easy | 2865 |

### Phase

| value | count |
| --- | --- |
| middlegame | 11854 |
| opening | 5208 |
| endgame | 4243 |

### Eval Bucket

| value | count |
| --- | --- |
| clear_black | 4615 |
| clear_white | 4366 |
| equal | 3482 |
| slight_white | 3449 |
| slight_black | 3441 |
| winning_black | 576 |
| winning_white | 568 |
| crushing_white | 412 |
| crushing_black | 396 |

### Tactical Motifs

| value | count |
| --- | --- |
| hanging | 11278 |
| fork | 7317 |
| pin | 5179 |
| (none) | 4308 |
| skewer | 4272 |
| overload | 1792 |
| discovered_attack | 1533 |
| mate_in_1 | 966 |
| promotion | 578 |
| underpromotion | 578 |

### Tag Families

| value | count |
| --- | --- |
| DEVELOPMENT | 21305 |
| FACT | 21305 |
| INITIATIVE | 21305 |
| KING | 21305 |
| MATERIAL | 21305 |
| META | 21305 |
| MOBILITY | 21305 |
| PAWN | 21305 |
| PIECE | 21305 |
| SPACE | 21305 |
| TACTIC | 16997 |
| ENDGAME | 6747 |
| OUTPOST | 3794 |
| THREAT | 578 |

## test

| field | value |
| --- | --- |
| rows | 21501 |
| tagged_rows | 21501 |
| rows_with_tactic_motif | 17279 |

### Difficulty

| value | count |
| --- | --- |
| very_hard | 5860 |
| medium | 5482 |
| hard | 4181 |
| easy | 3190 |
| very_easy | 2788 |

### Phase

| value | count |
| --- | --- |
| middlegame | 11835 |
| opening | 5328 |
| endgame | 4338 |

### Eval Bucket

| value | count |
| --- | --- |
| clear_black | 4757 |
| clear_white | 4497 |
| slight_black | 3488 |
| equal | 3485 |
| slight_white | 3406 |
| winning_white | 557 |
| winning_black | 552 |
| crushing_black | 404 |
| crushing_white | 355 |

### Tactical Motifs

| value | count |
| --- | --- |
| hanging | 11597 |
| fork | 7490 |
| pin | 5177 |
| skewer | 4277 |
| (none) | 4222 |
| overload | 1858 |
| discovered_attack | 1651 |
| mate_in_1 | 1050 |
| promotion | 632 |
| underpromotion | 632 |

### Tag Families

| value | count |
| --- | --- |
| DEVELOPMENT | 21501 |
| FACT | 21501 |
| INITIATIVE | 21501 |
| KING | 21501 |
| MATERIAL | 21501 |
| META | 21501 |
| MOBILITY | 21501 |
| PAWN | 21501 |
| PIECE | 21501 |
| SPACE | 21501 |
| TACTIC | 17279 |
| ENDGAME | 6807 |
| OUTPOST | 3753 |
| THREAT | 632 |
