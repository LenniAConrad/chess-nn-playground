# Dykstra-LCP Report Template

Required comparisons:

- LC0 BT4 puzzle_binary baseline on the same split;
- VetoSelect v2/A3 current best result;
- Dykstra-LCP aggregate metrics;
- matched-recall false positives, especially source fine label `1`;
- projection-distance, trace-residual, and slack means by source fine label.

Required benchmark reporting from `ideas/all_ideas/docs/BENCHMARK_REPORTING.md`:

- aggregate metrics plus fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`;
- highest-confidence wrong examples with FEN, difficulty, phase, and motifs;
- a short conclusion stating whether projection feasibility appears to add signal.
