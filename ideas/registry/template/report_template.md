# Idea Report Template

- Extra report sections:
- Required comparisons:
- Known blockers:

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an aggregate confusion matrix. Every promoted idea must require:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`, `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for fine label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and motifs;
- a short conclusion describing what the model appears able and unable to learn.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
- Slices where this idea is expected to fail:
- Ablation that should erase the slice-level gain:
- Minimum useful slice-level improvement:
