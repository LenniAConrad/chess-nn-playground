# Trainer Notes

## Trainer entry point

`ideas/registry/p037_gibbs_cut_log_partition/train.py` calls
`idea_train_cli(__file__)` which dispatches to the standard puzzle
binary trainer.

## Training contract

- `mode: puzzle_binary`
- `loss: bce_with_logits`
- `primary_metric: pr_auc`
- Same split as i193 (`crtk_sample_3class_unique_crtk_tags`).
- Seed `42` for the first scout run.

## Inputs not consumed

The trainer passes only the simple_18 board tensor `batch["x"]` into
the model. The cut-grid edge costs and source/sink penalties are
projected from the trunk joint feature -- they are not consumed from
CRTK metadata or any external feature.

## Validation order

1. `python scripts/train_model.py --list-models` -- expect
   `gibbs_cut_log_partition` to be listed.
2. `python scripts/validate_training_config.py --static
   ideas/registry/p037_gibbs_cut_log_partition/config.yaml`.
3. `python -m pytest tests/test_gibbs_cut_log_partition.py`.
4. `RUN_PRIMITIVE_TRAIN=0 RUN_PRIMITIVE_DRY_RUN=1
   ./run_primitive_pipeline.sh` -- the dry-run plan must include
   p037's config.

Scout training is *not* launched in this worktree because
`CLAUDE_ALLOW_TRAINING=0` is set by the launcher.

## Expected scout command

When `CLAUDE_ALLOW_TRAINING=1` is set:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
  --config ideas/registry/p037_gibbs_cut_log_partition/config.yaml
```

The matched-ablation control is the same command with
`model.ablation: shuffle_logpartition` injected into the model
section.

## Reporting

After a scout run completes:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reports/report_prediction_slices.py \
  --run-dir results/<run_dir> --splits val test
PYTHONDONTWRITEBYTECODE=1 python scripts/compare_results.py
```

`gibbs_log_partition_mean` and `gibbs_cut_edge_energy` are the
primitive's declared diagnostic axes; the keep/drop decision is whether
the unablated run improves the king-safety / fortress slice and whether
`shuffle_logpartition` loses most of that lift.
