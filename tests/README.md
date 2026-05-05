# Tests

The tests are intentionally kept as one flat pytest suite so simple commands keep working:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

The main coverage groups are:

- benchmark contracts: config paths, suite contents, canonical split usage, and model output contracts;
- data and encoding tests: FEN parsing, board features, JSON loading, datasets, and split helpers;
- idea tests: registry integrity, generated prompts, idea reporting, and representative idea model behavior;
- training/reporting tests: smoke training, artifact validation, training plots, result comparison, and speed-control defaults;
- system helper tests: local USB mount candidate detection.

Keep new tests focused and cheap. Long training belongs in scripts or run directories, not in pytest.
