# CRTK Tagged Split Report

These tags are metadata for benchmarking and error analysis only. They must not be used as neural-network input features.

## Files

- `data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet`
- `data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet`
- `data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet`

## train

| field | value |
| --- | --- |
| rows | 360000 |
| tagged_rows | 360000 |
| rows_with_tactic_motif | 288849 |

### Difficulty

| value | count |
| --- | --- |
| very_hard | 98187 |
| medium | 89426 |
| hard | 72929 |
| easy | 51870 |
| very_easy | 47588 |

### Phase

| value | count |
| --- | --- |
| middlegame | 199018 |
| opening | 88986 |
| endgame | 71996 |

### Eval Bucket

| value | count |
| --- | --- |
| clear_black | 77601 |
| clear_white | 75549 |
| equal | 59600 |
| slight_black | 58461 |
| slight_white | 56836 |
| winning_black | 9655 |
| winning_white | 9079 |
| crushing_black | 6739 |
| crushing_white | 6480 |

### Tactical Motifs

| value | count |
| --- | --- |
| hanging | 193300 |
| fork | 126155 |
| pin | 87940 |
| skewer | 73045 |
| (none) | 71151 |
| overload | 30394 |
| discovered_attack | 27593 |
| mate_in_1 | 17166 |
| promotion | 9959 |
| underpromotion | 9959 |

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
| TACTIC | 288849 |
| ENDGAME | 114186 |
| OUTPOST | 64858 |
| THREAT | 9959 |

## val

| field | value |
| --- | --- |
| rows | 45000 |
| tagged_rows | 45000 |
| rows_with_tactic_motif | 36025 |

### Difficulty

| value | count |
| --- | --- |
| very_hard | 12283 |
| medium | 11338 |
| hard | 9170 |
| easy | 6266 |
| very_easy | 5943 |

### Phase

| value | count |
| --- | --- |
| middlegame | 24868 |
| opening | 11020 |
| endgame | 9112 |

### Eval Bucket

| value | count |
| --- | --- |
| clear_black | 9644 |
| clear_white | 9321 |
| slight_black | 7494 |
| equal | 7420 |
| slight_white | 7166 |
| winning_black | 1175 |
| winning_white | 1158 |
| crushing_black | 829 |
| crushing_white | 793 |

### Tactical Motifs

| value | count |
| --- | --- |
| hanging | 24143 |
| fork | 15809 |
| pin | 10784 |
| skewer | 9112 |
| (none) | 8975 |
| overload | 3743 |
| discovered_attack | 3402 |
| mate_in_1 | 1997 |
| promotion | 1234 |
| underpromotion | 1234 |

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
| TACTIC | 36025 |
| ENDGAME | 14442 |
| OUTPOST | 8168 |
| THREAT | 1234 |

## test

| field | value |
| --- | --- |
| rows | 45000 |
| tagged_rows | 45000 |
| rows_with_tactic_motif | 36043 |

### Difficulty

| value | count |
| --- | --- |
| very_hard | 12151 |
| medium | 11516 |
| hard | 9053 |
| easy | 6448 |
| very_easy | 5832 |

### Phase

| value | count |
| --- | --- |
| middlegame | 24918 |
| opening | 11050 |
| endgame | 9032 |

### Eval Bucket

| value | count |
| --- | --- |
| clear_black | 9774 |
| clear_white | 9418 |
| slight_black | 7378 |
| equal | 7376 |
| slight_white | 7085 |
| winning_black | 1178 |
| winning_white | 1136 |
| crushing_black | 890 |
| crushing_white | 765 |

### Tactical Motifs

| value | count |
| --- | --- |
| hanging | 24190 |
| fork | 15648 |
| pin | 10830 |
| skewer | 9054 |
| (none) | 8957 |
| overload | 3790 |
| discovered_attack | 3439 |
| mate_in_1 | 2077 |
| promotion | 1211 |
| underpromotion | 1211 |

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
| TACTIC | 36043 |
| ENDGAME | 14363 |
| OUTPOST | 8179 |
| THREAT | 1211 |
