# Idea Report Template

- Extra report sections:
  - **CAIO fingerprint diagnostics**: histograms of
    `caio_constructive_mean`, `caio_destructive_mean`, `caio_curl_mean`,
    `caio_conjugacy_error`, `caio_amplitude_norm` over the val and test
    splits, grouped by `crtk_eval_bucket` and `crtk_difficulty`. Use the
    standard `predictions_<split>.parquet` columns surfaced by the
    trainer.
  - **Constructive vs destructive ratio by slice**: per
    `crtk_tactic_motifs` bucket, the ratio
    `caio_constructive_mean / (caio_constructive_mean + caio_destructive_mean)`.
    Pins / forks / mating-net puzzles should skew constructive; near-puzzle
    / quiet positions should skew destructive or near-balanced.
  - **Gate behaviour**: distribution of `primitive_gate` by slice. The
    gate should rise on slices where the primitive is helpful and stay
    near zero elsewhere.

- Required comparisons:
  - i193 baseline on the same split, seed, and training protocol.
  - This idea (`none` ablation) at the same scale.
  - At minimum the `real_only` (A1), `random_phase` (A2), and
    `shuffle_relation_masks` (A4) ablations on the same seed for the
    falsifier decision.

- Known blockers:
  - Complex-tensor backward + AMP combinations are kept in `float32` for
    safety; this is the documented compromise.
  - `torch.compile` is intentionally disabled for the first scout run;
    revisit once an empirical decision is in hand.

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
  - Near-puzzle false-positive rate at recall 0.80 (CAIO spec's primary
    declared target).
  - `crtk_eval_bucket = equal`, `hard`, `mate_in_1`, promotion /
    underpromotion buckets.
- Slices where this idea is expected to fail:
  - Pure stalemate / quiet positions (no phase structure to exploit).
- Ablation that should erase the slice-level gain:
  - `real_only` (A1) — replaces complex amplitudes with real ones.
  - `random_phase` (A2) — destroys learned phase.
  - `shuffle_relation_masks` (A4) — destroys relation structure.
- Minimum useful slice-level improvement: at least `0.01` absolute
  improvement in matched-recall near-puzzle FP rate at recall 0.80.
