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

## Per-Stream Diagnostics

- Mechanism family: `king_path`
- Per-stream main logits (`exchange_logit`, `king_logit`, `positional_logit`):
- Mixture weights (`alpha_exchange`, `alpha_king`, `alpha_positional`):
- Residual logit (`residual_logit`):
- Route entropy (`route_entropy`):
- Stream disagreement (`stream_disagreement`):
- Per-stream pool norms (`exchange_pool_norm`, `king_pool_norm`, `positional_pool_norm`):
- Per-stream aux logits (`exchange_aux_logit`, `king_aux_logit`, `positional_aux_logit`):
- Mechanism energy (`mechanism_energy`):
- Multistream ablation code (`multistream_ablation`):
- Near-puzzle false positives:

## Ablation Deltas

- `none` vs `no_chess_bias`:
- `none` vs `no_phase_router`:
- `none` vs `remove_positional_stream`:
- `none` vs `remove_king_stream`:
- `none` vs `remove_exchange_stream`:
- `none` vs `no_aux_heads`:

## Slice Findings

Summarize performance by:

- `crtk_difficulty`
- `crtk_phase`
- `crtk_eval_bucket`
- `crtk_tactic_motifs`
- `crtk_tag_families`

## Decision

State whether `Multi-Stream Chess-Decomposed Transformer Evaluator` is kept, refined, scaled, or rejected. The decision must cite both aggregate metrics and slice behavior, including whether the `no_chess_bias` central falsifier and the per-stream removal ablations were beaten.
