# VetoSelect Report Template

Record:

- aggregate PR AUC, F1, precision, recall, confusion matrix;
- source-class 3x2 diagnostics, especially near-puzzle false positives;
- `selector_logit` and reject-rate behavior for true puzzles versus near-puzzle negatives;
- required slice analysis from `ideas/BENCHMARK_REPORTING.md`, including `slice_report_val.md`, `slice_report_test.md`, `crtk_difficulty`, `crtk_phase`, eval-bucket, motif, and tag-family slice strengths/weaknesses;
- comparison to the latest LC0 BT4 binary baseline on the same split.
