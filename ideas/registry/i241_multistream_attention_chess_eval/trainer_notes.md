# Trainer Notes

Use the guarded idea `train.py`. The config is scout-tier, CUDA-required, and uses the canonical tagged CRTK split. New runs must pass `scripts/validate_run_artifacts.py`.

The bespoke `multistream_attention_chess_eval` model returns a single puzzle logit plus per-stream and routing diagnostics. Run the seven `ablation` modes alongside the default `none` config to populate the falsification deltas in `report_template.md`.

The model exposes per-stream auxiliary logits (`exchange_aux_logit`, `king_aux_logit`, `positional_aux_logit`) and a configurable broadcast `aux_loss_weight` scalar; the puzzle_binary trainer does not consume the aux losses, but the diagnostics are recorded for downstream scaled-engine training plans.
