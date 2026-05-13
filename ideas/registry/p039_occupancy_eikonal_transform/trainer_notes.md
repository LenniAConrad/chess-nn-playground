# Trainer Notes

## Trainer entry point

`ideas/registry/p039_occupancy_eikonal_transform/train.py` calls
`idea_train_cli(__file__)`.

## Training contract

- `mode: puzzle_binary`
- `loss: bce_with_logits`
- `primary_metric: pr_auc`
- Same split as i193 (`crtk_sample_3class_unique_crtk_tags`).
- Seed `42` for the first scout run.

## Validation order

1. `python scripts/train_model.py --list-models` -- expect
   `occupancy_eikonal_transform` to be listed.
2. `python scripts/validate_training_config.py --static
   ideas/registry/p039_occupancy_eikonal_transform/config.yaml`.
3. `python -m pytest tests/test_occupancy_eikonal_transform.py`.
4. `RUN_PRIMITIVE_TRAIN=0 RUN_PRIMITIVE_DRY_RUN=1
   ./run_primitive_pipeline.sh`.

Scout training is *not* launched in this worktree because
`CLAUDE_ALLOW_TRAINING=0`.

## Expected scout command

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
  --config ideas/registry/p039_occupancy_eikonal_transform/config.yaml
```

## Reporting

`eikonal_field_range` is the primitive's key diagnostic axis. Slice
the validation/test set by `eikonal_field_range` percentile and check
whether the unablated run improves the high-range bucket (sharp
distance distinctions) while `shuffle_field` loses that lift.
