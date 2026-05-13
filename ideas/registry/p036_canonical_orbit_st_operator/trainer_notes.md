# Trainer Notes

## Trainer entry point

`ideas/registry/p036_canonical_orbit_st_operator/train.py` calls
`idea_train_cli(__file__)` which dispatches to the standard puzzle
binary trainer. The trainer reads `config.yaml` from the same folder.

## Training contract

- `mode: puzzle_binary`
- `loss: bce_with_logits`
- `primary_metric: pr_auc`
- Same split as i193 (`crtk_sample_3class_unique_crtk_tags`).
- Seed `42` for the first scout run.
- Mixed precision enabled; `allow_tf32: true`.
- Early stopping patience `5` and `min_epochs: 10`, matching i193.

## Inputs not consumed

The trainer passes only the simple_18 board tensor `batch["x"]` into
the model. CRTK tags and FEN are retained for the slice reports but are
*not* observed by the model. The orbit canonicalisation is computed in
latent space from the trunk's projected joint feature.

## Validation order

1. `python scripts/train_model.py --list-models` -- expect
   `canonical_orbit_st_operator` to be listed.
2. `python scripts/validate_training_config.py --static
   ideas/registry/p036_canonical_orbit_st_operator/config.yaml`.
3. `python -m pytest tests/test_canonical_orbit_st_operator.py`.
4. `RUN_PRIMITIVE_TRAIN=0 RUN_PRIMITIVE_DRY_RUN=1
   ./run_primitive_pipeline.sh` -- the dry-run plan must include
   `p036_canonical_orbit_st_operator/config.yaml`.

Scout training is *not* launched in this worktree because
`CLAUDE_ALLOW_TRAINING=0` is set by the launcher.

## Expected scout command

When `CLAUDE_ALLOW_TRAINING=1` is set, the scout run is:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
  --config ideas/registry/p036_canonical_orbit_st_operator/config.yaml
```

The matched-ablation control is the same command with
`model.ablation: shuffle_canonical` (or `identity_only`) injected into
the model section. Both runs must use the same split, seed, and
training budget as the baseline i193 control.

## Reporting

After a scout run completes:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reports/report_prediction_slices.py \
  --run-dir results/<run_dir> --splits val test
PYTHONDONTWRITEBYTECODE=1 python scripts/compare_results.py
```

`cost_orbit_gap` and `cost_orbit_ties` are the primitive's declared
slices; the keep/drop decision is whether the unablated run improves
those slices and whether `shuffle_canonical` loses most of that lift.
