# crtk Sample Split Report

- Input: `data/processed/crtk_usb_training_fast.parquet`
- Output dir: `data/splits/crtk_sample_3class_unique`
- Mode: `fine_3class`
- Label column: `fine_label`
- Max per class requested: `150,000`
- Rows written: `215,835`
- Elapsed minutes: `3.15`
- Batch size: `200,000`
- De-duplicate normalized FEN: `True`

## Rows By Split/Class

```json
{
  "train": {
    "0": 57971,
    "1": 57519,
    "2": 57539
  },
  "val": {
    "0": 7189,
    "1": 7012,
    "2": 7104
  },
  "test": {
    "0": 7206,
    "1": 7240,
    "2": 7055
  }
}
```

## Memory Safety

- The script streams Parquet batches and never loads the full 45M-row file.
- Output size is capped by `--max-per-class` so the current pandas-based trainer can open the split files.
- Split assignment is deterministic from `split_group_id`, falling back to FEN, to avoid group leakage.
- When `--dedupe-normalized-fen` is enabled, each normalized FEN can appear at most once across all splits.

## Warnings

- Only wrote 57,971/120,000 rows for split=train, class=0. Increase --sample-multiplier or lower --max-per-class.
- Only wrote 57,519/120,000 rows for split=train, class=1. Increase --sample-multiplier or lower --max-per-class.
- Only wrote 57,539/120,000 rows for split=train, class=2. Increase --sample-multiplier or lower --max-per-class.
- Only wrote 7,189/15,000 rows for split=val, class=0. Increase --sample-multiplier or lower --max-per-class.
- Only wrote 7,012/15,000 rows for split=val, class=1. Increase --sample-multiplier or lower --max-per-class.
- Only wrote 7,104/15,000 rows for split=val, class=2. Increase --sample-multiplier or lower --max-per-class.
- Only wrote 7,206/15,000 rows for split=test, class=0. Increase --sample-multiplier or lower --max-per-class.
- Only wrote 7,240/15,000 rows for split=test, class=1. Increase --sample-multiplier or lower --max-per-class.
- Only wrote 7,055/15,000 rows for split=test, class=2. Increase --sample-multiplier or lower --max-per-class.