# Contributing

Thanks for taking the time to improve Chess NN Playground. This repository is a
research harness, so changes should preserve benchmark comparability, artifact
contracts, and clear provenance.

## License Notice

This repository is currently all rights reserved. Opening an issue or pull
request does not grant permission to use the repository outside the rights stated
in `LICENSE`.

By submitting a contribution, you represent that you have the right to submit it
and that the repository owner may review, modify, and incorporate it into this
repository. Do not submit third-party code, data, model weights, generated
outputs, or documentation unless you can identify the source and license terms.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Use the packaged console commands instead of path-bootstrapped script execution:

```bash
chess-nn-train --list-models
chess-nn-validate-config --static configs/benchmarks/puzzle_binary/bench_lc0_bt4_classifier.yaml
```

## Required Checks

Run the CPU CI gate before opening a pull request:

```bash
ruff check .
python -m compileall -q src scripts tests
python -m pytest \
  tests/test_compare_results.py \
  tests/test_run_artifact_validation.py::test_legacy_run_artifacts_are_warnings_only_when_allowed \
  tests/test_flop_report.py \
  tests/test_paper_report.py \
  tests/test_paper_ready_runner.py \
  tests/test_training_speed_controls.py \
  tests/test_research_packet_promotions.py \
  tests/test_config_validation.py
python -m pytest tests/test_training_smoke.py
```

The full pytest suite is still useful, but it currently includes known registry,
report-template, and historical-result backlog failures. Do not hide new
failures behind those known failures; call out any difference from the current
baseline in the pull request.

## Change Guidelines

- Keep benchmark data contracts stable. Do not change label mapping, split
  semantics, CRTK reporting fields, or artifact expectations without updating
  docs and tests.
- Keep model changes registered through the shared package APIs and standard
  trainer artifacts.
- Do not commit local datasets, checkpoints, model weights, secrets, environment
  files, or machine-specific paths.
- Regenerate generated indexes with `chess-nn-build-idea-catalog` instead of
  hand-editing them.
- Prefer small pull requests with one clear behavioral objective.
- Include validation commands and known residual risks in the pull request body.

## Pull Request Review Expectations

A review should focus on correctness, reproducibility, benchmark comparability,
artifact coverage, and whether the implementation fits the existing package
structure. Cosmetic-only refactors should be kept separate from benchmark or
training behavior changes.

