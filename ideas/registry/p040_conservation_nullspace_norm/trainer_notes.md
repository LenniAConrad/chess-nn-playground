# Trainer Notes

## Trainer entry point

`ideas/registry/p040_conservation_nullspace_norm/train.py` calls
`idea_train_cli(__file__)`.

## Training contract

- `mode: puzzle_binary`
- `loss: bce_with_logits`
- `primary_metric: pr_auc`
- Same split as i193 (`crtk_sample_3class_unique_crtk_tags`).
- Seed `42` for the first scout run.

## Validation order

1. `python scripts/train_model.py --list-models` -- expect
   `conservation_nullspace_norm` to be listed.
2. `python scripts/validate_training_config.py --static
   ideas/registry/p040_conservation_nullspace_norm/config.yaml`.
3. `python -m pytest tests/test_conservation_nullspace_norm.py`.
4. `RUN_PRIMITIVE_TRAIN=0 RUN_PRIMITIVE_DRY_RUN=1
   ./run_primitive_pipeline.sh`.

Scout training is *not* launched in this worktree because
`CLAUDE_ALLOW_TRAINING=0`.

## Expected scout command

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
  --config ideas/registry/p040_conservation_nullspace_norm/config.yaml
```

## Reporting

`cnnorm_explained_frac` is the primitive's key diagnostic axis. Slice
the validation / test set by `cnnorm_explained_frac` percentile and
check whether the unablated run improves the high-fraction bucket
while `shuffle_residual` and `no_projection` lose that lift.
