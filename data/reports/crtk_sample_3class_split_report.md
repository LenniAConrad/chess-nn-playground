# crtk Sample Split Report

- Input: `data/processed/crtk_training_20260419_180229_fast.parquet`
- Output dir: `data/splits/crtk_sample_3class`
- Mode: `fine_3class`
- Label column: `fine_label`
- Max per class requested: `150,000`
- Rows written: `450,000`
- Elapsed minutes: `2.08`
- Batch size: `200,000`

## Rows By Split/Class

```json
{
  "train": {
    "0": 120000,
    "1": 120000,
    "2": 120000
  },
  "val": {
    "0": 15000,
    "1": 15000,
    "2": 15000
  },
  "test": {
    "0": 15000,
    "1": 15000,
    "2": 15000
  }
}
```

## Memory Safety

- The script streams Parquet batches and never loads the full 45M-row file.
- Output size is capped by `--max-per-class` so the current pandas-based trainer can open the split files.
- Split assignment is deterministic from `split_group_id`, falling back to FEN, to avoid group leakage.
