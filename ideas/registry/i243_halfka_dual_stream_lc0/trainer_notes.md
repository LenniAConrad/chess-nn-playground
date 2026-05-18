# Trainer Notes

Use the guarded idea `train.py`. The config is scout-tier, CUDA-required, and uses the canonical tagged CRTK split. New runs must pass `scripts/validate_run_artifacts.py`.

The bespoke `halfka_dual_stream_lc0` model returns a single puzzle logit plus per-stream, HalfKA-accumulator, and LC0-head diagnostics. Run the five `ablation` modes alongside the default `none` config to populate the falsification deltas in `report_template.md`.

The LC0 value (`value_wdl_logits`) and policy (`policy_logits`) heads are exposed as diagnostics; the puzzle_binary trainer ignores them. Engine-grade Stockfish-eval distillation or LC0 self-play training pipelines that would consume those heads are out of scope for this implementation.
