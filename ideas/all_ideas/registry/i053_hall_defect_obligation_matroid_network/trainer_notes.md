# Trainer Notes

Use the guarded idea `train.py`. The model follows the repo's puzzle-binary BCE contract: fine labels `0` and `1` map to non-puzzle, fine label `2` maps to puzzle, and forward returns `output["logits"]` with shape `(B,)`.

The deterministic Hall branch is derived only from the current board tensor. It does not use CRTK/source metadata, engine values, legal move counts, or future positions. Diagnostic outputs should be saved with predictions for Hall-defect ablation analysis.

The config remains paper-grade, CUDA-required, and uses the canonical tagged CRTK split. New runs must pass `scripts/validate_run_artifacts.py`.
