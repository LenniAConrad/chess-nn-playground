# Trainer Notes

Use the guarded idea `train.py`. The config is paper-grade, CUDA-required, and uses the canonical tagged CRTK split. New runs must pass `scripts/validate_run_artifacts.py`.

The bespoke `bispectral_phase_coupling_board_network` model returns a single puzzle logit plus bispectral diagnostics. Run the seven `ablation` modes alongside the default `none` config to populate the falsification deltas in `report_template.md`.
