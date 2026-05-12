# Trainer Notes

Use the guarded idea entrypoint:

```bash
python ideas/all_ideas/registry/i013_sparse_relation_pursuit_asymmetry/train.py
```

For benchmark-suite execution:

```bash
python scripts/run_experiment_suite.py configs/bench_srpa_lc0bt4.yaml --jobs 1
```

The config requires `device: nvidia`, uses the canonical CRTK tagged split, enables AMP/TF32, and trains with `training.loss: srpa`.

Paper-grade promotion requires at least seeds `42`, `43`, and `44`, `epochs >= 20`, `min_epochs >= 10`, validation plateau scheduling, full artifact validation, and slice reports under the standard benchmark reporting rules.
