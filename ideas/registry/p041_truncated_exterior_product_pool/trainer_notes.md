# Trainer Notes

## Trainer entry point

`ideas/registry/p041_truncated_exterior_product_pool/train.py` calls
`idea_train_cli(__file__)`.

## Training contract

- `mode: puzzle_binary`
- `loss: bce_with_logits`
- `primary_metric: pr_auc`
- Same split as i193 (`crtk_sample_3class_unique_crtk_tags`).
- Seed `42` for the first scout run.

## Validation order

1. `python scripts/train_model.py --list-models` -- expect
   `truncated_exterior_product_pool` to be listed.
2. `python scripts/validate_training_config.py --static
   ideas/registry/p041_truncated_exterior_product_pool/config.yaml`.
3. `python -m pytest tests/test_truncated_exterior_product_pool.py`.
4. `RUN_PRIMITIVE_TRAIN=0 RUN_PRIMITIVE_DRY_RUN=1
   ./run_primitive_pipeline.sh`.

Scout training is *not* launched in this worktree because
`CLAUDE_ALLOW_TRAINING=0`.

## Expected scout command

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
  --config ideas/registry/p041_truncated_exterior_product_pool/config.yaml
```

## Reporting

`tepp_grade_2_magnitude` is the primitive's key diagnostic axis (when
`max_grade >= 2`). Slice the validation / test set by
`tepp_grade_2_magnitude` percentile and check whether the unablated
run improves the high-magnitude bucket (positions where the wedge
cancellation is informative) while `first_order_only` and
`shuffle_grades_high` lose that lift.
