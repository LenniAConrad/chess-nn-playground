# Trainer Notes

## Trainer entry point

`ideas/registry/p038_woodbury_set_resolver/train.py` calls
`idea_train_cli(__file__)` which dispatches to the standard puzzle
binary trainer.

## Training contract

- `mode: puzzle_binary`
- `loss: bce_with_logits`
- `primary_metric: pr_auc`
- Same split as i193 (`crtk_sample_3class_unique_crtk_tags`).
- Seed `42` for the first scout run.

## Validation order

1. `python scripts/train_model.py --list-models` -- expect
   `woodbury_set_resolver` to be listed.
2. `python scripts/validate_training_config.py --static
   ideas/registry/p038_woodbury_set_resolver/config.yaml`.
3. `python -m pytest tests/test_woodbury_set_resolver.py`.
4. `RUN_PRIMITIVE_TRAIN=0 RUN_PRIMITIVE_DRY_RUN=1
   ./run_primitive_pipeline.sh`.

Scout training is *not* launched in this worktree because
`CLAUDE_ALLOW_TRAINING=0`.

## Expected scout command

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
  --config ideas/registry/p038_woodbury_set_resolver/config.yaml
```

## Reporting

`wsr_logdet_A`, `wsr_leverage_mean`, and `wsr_leverage_max` are the
primitive's declared diagnostic axes. Slice the validation/test set by
leverage-variance percentile and check whether the unablated run
improves the high-variance bucket while `shuffle_active_tokens` loses
that lift.
