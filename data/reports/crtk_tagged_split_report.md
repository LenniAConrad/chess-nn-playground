# CRTK Tagged Split Report

These tags are metadata for benchmarking and error analysis only. They must not be used as neural-network input features.

## Files

- `data/splits/crtk_sample_3class_crtk_tags/split_train.parquet`
- `data/splits/crtk_sample_3class_crtk_tags/split_val.parquet`
- `data/splits/crtk_sample_3class_crtk_tags/split_test.parquet`

## train

| field | value |
| --- | --- |
| rows | 360000 |
| tagged_rows | 360000 |
| rows_with_tactic_motif | 287968 |

### Difficulty

| value | count |
| --- | --- |
| very_hard | 97853 |
| medium | 89619 |
| hard | 73185 |
| easy | 52053 |
| very_easy | 47290 |

### Phase

| value | count |
| --- | --- |
| middlegame | 197143 |
| opening | 90032 |
| endgame | 72825 |

### Eval Bucket

| value | count |
| --- | --- |
| clear_black | 77453 |
| clear_white | 75586 |
| equal | 59735 |
| slight_black | 58697 |
| slight_white | 56887 |
| winning_black | 9554 |
| winning_white | 8941 |
| crushing_black | 6704 |
| crushing_white | 6443 |

### Tactical Motifs

| value | count |
| --- | --- |
| hanging | 192708 |
| fork | 125740 |
| pin | 87604 |
| skewer | 73003 |
| (none) | 72032 |
| overload | 30257 |
| discovered_attack | 27399 |
| mate_in_1 | 17336 |
| promotion | 10200 |
| underpromotion | 10200 |

### Tag Families

| value | count |
| --- | --- |
| DEVELOPMENT | 360000 |
| FACT | 360000 |
| INITIATIVE | 360000 |
| KING | 360000 |
| MATERIAL | 360000 |
| META | 360000 |
| MOBILITY | 360000 |
| PAWN | 360000 |
| PIECE | 360000 |
| SPACE | 360000 |
| TACTIC | 287968 |
| ENDGAME | 114542 |
| OUTPOST | 64156 |
| THREAT | 10200 |

## val

| field | value |
| --- | --- |
| rows | 45000 |
| tagged_rows | 45000 |
| rows_with_tactic_motif | 35951 |

### Difficulty

| value | count |
| --- | --- |
| very_hard | 12258 |
| medium | 11305 |
| hard | 9259 |
| easy | 6286 |
| very_easy | 5892 |

### Phase

| value | count |
| --- | --- |
| middlegame | 24649 |
| opening | 11129 |
| endgame | 9222 |

### Eval Bucket

| value | count |
| --- | --- |
| clear_black | 9698 |
| clear_white | 9265 |
| slight_black | 7485 |
| equal | 7451 |
| slight_white | 7180 |
| winning_black | 1184 |
| winning_white | 1117 |
| crushing_black | 828 |
| crushing_white | 792 |

### Tactical Motifs

| value | count |
| --- | --- |
| hanging | 24074 |
| fork | 15768 |
| pin | 10704 |
| skewer | 9111 |
| (none) | 9049 |
| overload | 3750 |
| discovered_attack | 3421 |
| mate_in_1 | 2006 |
| promotion | 1261 |
| underpromotion | 1261 |

### Tag Families

| value | count |
| --- | --- |
| DEVELOPMENT | 45000 |
| FACT | 45000 |
| INITIATIVE | 45000 |
| KING | 45000 |
| MATERIAL | 45000 |
| META | 45000 |
| MOBILITY | 45000 |
| PAWN | 45000 |
| PIECE | 45000 |
| SPACE | 45000 |
| TACTIC | 35951 |
| ENDGAME | 14495 |
| OUTPOST | 8108 |
| THREAT | 1261 |

## test

| field | value |
| --- | --- |
| rows | 45000 |
| tagged_rows | 45000 |
| rows_with_tactic_motif | 35990 |

### Difficulty

| value | count |
| --- | --- |
| very_hard | 12109 |
| medium | 11436 |
| hard | 9275 |
| easy | 6389 |
| very_easy | 5791 |

### Phase

| value | count |
| --- | --- |
| middlegame | 24556 |
| opening | 11287 |
| endgame | 9157 |

### Eval Bucket

| value | count |
| --- | --- |
| clear_black | 9708 |
| clear_white | 9323 |
| equal | 7507 |
| slight_black | 7385 |
| slight_white | 7149 |
| winning_black | 1175 |
| winning_white | 1130 |
| crushing_black | 865 |
| crushing_white | 758 |

### Tactical Motifs

| value | count |
| --- | --- |
| hanging | 24160 |
| fork | 15742 |
| pin | 10814 |
| (none) | 9010 |
| skewer | 8984 |
| overload | 3722 |
| discovered_attack | 3429 |
| mate_in_1 | 2150 |
| promotion | 1260 |
| underpromotion | 1260 |

### Tag Families

| value | count |
| --- | --- |
| DEVELOPMENT | 45000 |
| FACT | 45000 |
| INITIATIVE | 45000 |
| KING | 45000 |
| MATERIAL | 45000 |
| META | 45000 |
| MOBILITY | 45000 |
| PAWN | 45000 |
| PIECE | 45000 |
| SPACE | 45000 |
| TACTIC | 35990 |
| ENDGAME | 14363 |
| OUTPOST | 8065 |
| THREAT | 1260 |
