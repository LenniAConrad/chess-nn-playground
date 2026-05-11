# crtk Fast Import Report

- Input: `/home/lennart/Documents/chess-rtk/data/exported/crtk_usb_training.jsonl`
- Output: `data/processed/crtk_usb_training_fast.parquet`
- Rows: `132,576,864`
- Elapsed minutes: `1.87`
- Rows/sec: `1,180,961`
- Compression: `snappy`

## Label-status distribution

- `known_non_puzzle`: 98,299,305
- `candidate_1_or_2_unresolved`: 0
- `verified_near_puzzle`: 28,752,203
- `verified_puzzle`: 5,525,356

## Notes

- FEN validation and derived board summaries were skipped for speed.
- Deduplication was skipped; identical FENs share `split_group_id = normalized_fen` when no stronger group exists.
- Engine metadata is retained only for selected scalar columns, not nested `multipv`.
