# Idea Report Template

- Extra report sections:
  - **DHPE fingerprint diagnostics**: histograms of `dhpe_z_pos`,
    `dhpe_z_neg`, `dhpe_z_ratio`, `dhpe_z_top1` over the val and test
    splits, grouped by `crtk_eval_bucket` and `crtk_difficulty`. Use the
    standard `predictions_<split>.parquet` columns surfaced by the
    trainer.
  - **Top-pair sign analysis**: per `crtk_tactic_motifs` bucket, the
    fraction of positions where `dhpe_z_pos > dhpe_z_neg` vs the inverse.
    Pin / fork / discovered-attack puzzles should skew super-additive;
    near-puzzle / defender-trap cases should skew sub-additive.
  - **Gate behaviour**: distribution of `primitive_gate` by slice. The
    gate should be near zero on quiet positions and rise on tactical
    slices where the primitive contributes.

- Required comparisons:
  - i193 baseline on the same split, seed, and training protocol.
  - This idea (`none` ablation) at the same scale.
  - At minimum the `unsigned` (A1) and `no_dhpe` (A2) ablations on the same
    seed for the falsifier decision.

- Known blockers:
  - Saliency stage is deterministic (piece-value prior). If the equal-slice
    lift is weak, the next experiment is to flip the saliency to a learned
    per-piece dropout, which costs another `top_k` PhiScorer forwards per
    position.

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an aggregate
confusion matrix. Every promoted idea must require:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for fine
  label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and motifs;
- a short conclusion describing what the model appears able and unable to
  learn.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  - `crtk_eval_bucket = equal` (DHPE spec's declared target).
  - `crtk_difficulty = hard` / near-puzzle.
  - Pin / fork / discovered-attack puzzles (positive-Hessian slices).
- Slices where this idea is expected to fail:
  - Pure mate-in-1 / king-safety positions (TSDP territory).
  - Promotion / underpromotion (PFCT territory).
- Ablation that should erase the slice-level gain:
  - `unsigned` (A1) — replaces signed Hessian with `|H|`. If the lift
    survives, the sign isn't load-bearing.
  - `shuffled_pairs` (A3) — destroys pair-identity. If the lift survives,
    pair structure isn't load-bearing.
- Minimum useful slice-level improvement: `>= 0.018 PR AUC` on the
  `equal` bucket vs the matched i193 baseline.
