# Trainer Notes — p007 Attack-Ray Sparse Attention

- Use `train.py` as-is; it calls `idea_train_cli(__file__)`.
- The shared `puzzle_binary` loss/dataloader/checkpointing/report
  pipeline is reused unchanged. No new dataset column or collate hook
  is required — the ray index is derived inside the forward pass from
  the simple_18 board.
- Smoke command:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
    tests/test_attack_ray_sparse_attention.py
  ```

- Static config validation:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
    scripts/validate_training_config.py --static \
    ideas/registry/p007_attack_ray_sparse_attention/config.yaml
  ```

- Scout training (only with `CLAUDE_ALLOW_TRAINING=1`):

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/train_model.py \
    --config ideas/registry/p007_attack_ray_sparse_attention/config.yaml
  ```

- Falsifier sweep:

  ```bash
  RUN_PRIMITIVE_TRAIN=1 RUN_PRIMITIVE_DRY_RUN=0 \
    RUN_PRIMITIVE_CONFIGS=ideas/registry/p007_attack_ray_sparse_attention/config.yaml \
    ./run_primitive_pipeline.sh
  ```

- Post-run reporting: `scripts/reports/report_prediction_slices.py
  --run-dir results/<dir> --splits val test`; record results in
  `report_template.md`.
