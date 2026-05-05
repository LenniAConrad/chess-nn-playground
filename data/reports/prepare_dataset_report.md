# Prepare Dataset Report

- Total raw rows: `6`
- Valid rows after dedup: `6`
- Invalid/rejected rows: `0`
- Duplicates removed: `0`
- Missing FEN count: `0`
- Output: `data/processed/smoke_positions.parquet`
- Rejected output: `data/processed/smoke_rejected_positions.parquet`

## Label-status distribution

- `known_non_puzzle`: 3
- `candidate_1_or_2_unresolved`: 3

## Validation

```json
{
  "valid": true,
  "missing_columns": [],
  "bad_label_status": [],
  "duplicate_normalized_fens": 0,
  "rows": 6
}
```
