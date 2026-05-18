# Trainer Notes

Use the guarded idea `train.py`. The config is paper-grade, CUDA-required, and uses the canonical tagged CRTK split. New runs must pass `scripts/validate_run_artifacts.py`.

The coverage head exposes per-attribute coverage, top marginal gains, and concept entropy in the prediction artifact; downstream slice reports can read those keys without changes.
