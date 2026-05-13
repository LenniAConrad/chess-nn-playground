# Trainer Notes — p008 Rule-Conditioned Sparse Attention (MobScan)

- Use `train.py` as-is.
- No new dataset column required; the adjacency is derived inside the
  forward pass from the simple_18 board.
- Smoke command:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
    tests/test_rule_conditioned_sparse_attention.py
  ```

- Static config validation:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
    scripts/validate_training_config.py --static \
    ideas/registry/p008_rule_conditioned_sparse_attention/config.yaml
  ```

- Scout training (only if `CLAUDE_ALLOW_TRAINING=1`):

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/train_model.py \
    --config ideas/registry/p008_rule_conditioned_sparse_attention/config.yaml
  ```

- Falsifier sweep:

  ```bash
  RUN_PRIMITIVE_TRAIN=1 RUN_PRIMITIVE_DRY_RUN=0 \
    RUN_PRIMITIVE_CONFIGS=ideas/registry/p008_rule_conditioned_sparse_attention/config.yaml \
    ./run_primitive_pipeline.sh
  ```
