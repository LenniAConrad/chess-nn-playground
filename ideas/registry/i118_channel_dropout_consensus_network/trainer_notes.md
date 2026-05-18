# Trainer Notes

Use the guarded idea `train.py`. The config is paper-grade, CUDA-required, and uses the canonical tagged CRTK split. New runs must pass `scripts/validate_run_artifacts.py`.

The model loads via the `channel_dropout_consensus_network` registry key and lives at `src/chess_nn_playground/models/trunk/channel_dropout_consensus.py`. Each forward pass stacks `V = 6` channel-dropped board views along the batch axis before invoking the shared trunk, so the effective compute and BatchNorm batch size of the encoder are `V * batch_size`. Keep this in mind when tuning `training.batch_size` against GPU memory: at `channels: 64`, `depth: 2`, `batch_size: 256`, the encoder sees a 1536-row batch per step.

For ablation runs, set `model.ablation` to one of the documented modes (`none`, `full_view_only`, `mean_only`, `random_channel_masks`, `train_dropout_only`). The `full_view_only` and `train_dropout_only` ablations skip the multi-view stacking and are roughly `V`x cheaper per step; `mean_only` and `random_channel_masks` keep the multi-view trunk and have the same per-step cost as `none`.
