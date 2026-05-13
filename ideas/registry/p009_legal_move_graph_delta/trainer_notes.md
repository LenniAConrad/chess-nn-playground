# Trainer Notes — p009 Legal-Move-Graph Convolution

- Use `train.py` as-is.
- No new dataset column required.
- Smoke command:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
    tests/test_legal_move_graph_delta.py
  ```

- Static config validation:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
    scripts/validate_training_config.py --static \
    ideas/registry/p009_legal_move_graph_delta/config.yaml
  ```

- Scout training (only if `CLAUDE_ALLOW_TRAINING=1`):

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/train_model.py \
    --config ideas/registry/p009_legal_move_graph_delta/config.yaml
  ```

- Falsifier sweep:

  ```bash
  RUN_PRIMITIVE_TRAIN=1 RUN_PRIMITIVE_DRY_RUN=0 \
    RUN_PRIMITIVE_CONFIGS=ideas/registry/p009_legal_move_graph_delta/config.yaml \
    ./run_primitive_pipeline.sh
  ```
