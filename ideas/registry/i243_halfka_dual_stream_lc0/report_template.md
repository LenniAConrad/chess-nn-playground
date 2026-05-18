# Report Template

## Run

- Result path:
- Config:
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## HalfKA + Dual-Stream Diagnostics

- Mechanism family: `king_path`
- Per-stream main logits (`exchange_logit`, `king_logit`):
- Phase-router gate (`alpha_king`, `alpha_exchange`):
- Residual logit (`residual_logit`):
- Per-stream pool norms (`exchange_pool_norm`, `king_pool_norm`):
- White / black HalfKA accumulator norms (`white_accumulator_norm`, `black_accumulator_norm`):
- Combined accumulator norm (`accumulator_norm`):
- LC0 head shapes (`value_wdl_logits` 3 logits, `policy_logits` 32 logits): exposed only as diagnostics under the puzzle_binary trainer.
- Mechanism energy (`mechanism_energy`):
- Halfka dual-stream ablation code (`halfka_dual_stream_ablation`):
- Near-puzzle false positives:

## Ablation Deltas

- `none` vs `no_halfka`:
- `none` vs `no_dual_stream`:
- `none` vs `no_residual`:
- `none` vs `puzzle_only`:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `HalfKA Dual-Stream LC0 Evaluator` is kept, refined, scaled, or rejected at puzzle_binary scout scale. The decision must cite both aggregate metrics and slice behavior, including whether the `no_halfka` and `no_dual_stream` central falsifiers were beaten. Engine-grade decisions (e.g. comparing to Stockfish NNUE elo) require the scaled variant and engine training pipeline, which are out of scope for this implementation.
