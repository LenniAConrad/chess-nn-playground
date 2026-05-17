# Tests

The tests are intentionally kept as one flat pytest suite so simple commands keep working:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

The CI gate uses a stable CPU subset while the full suite still has known registry/report/result backlog failures:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest \
  tests/test_compare_results.py \
  tests/test_run_artifact_validation.py::test_legacy_run_artifacts_are_warnings_only_when_allowed \
  tests/test_flop_report.py \
  tests/test_paper_report.py \
  tests/test_paper_ready_runner.py \
  tests/test_training_speed_controls.py \
  tests/test_research_packet_promotions.py \
  tests/test_config_validation.py \
  tests/test_training_smoke.py
```

The main coverage groups are:

- benchmark contracts: config paths, suite contents, canonical split usage, and model output contracts;
- data and encoding tests: FEN parsing, board features, JSON loading, datasets, and split helpers;
- idea tests: registry integrity, generated prompts, idea reporting, and representative idea model behavior;
- training/reporting tests: smoke training, artifact validation, training plots, result comparison, and speed-control defaults;
- system helper tests: local USB mount candidate detection.

Keep new tests focused and cheap. Long training belongs in scripts or run directories, not in pytest.
