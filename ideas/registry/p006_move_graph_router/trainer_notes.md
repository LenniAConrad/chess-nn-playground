# Trainer Notes — p006 Move-Graph Router

- Use `train.py` as-is. It calls `idea_train_cli(__file__)`, which checks
  idea/config/model identity before training.
- The guard requires `implementation_status` to be `implemented` or
  `tested`, `implementation_kind: bespoke_model`, `device: nvidia`, and a
  registered `model.name`. The config in this folder satisfies all four.
- The shared `puzzle_binary` loss/dataloader/checkpointing/report pipeline
  is reused unchanged. The primitive head does **not** require any extra
  trainer plumbing — it derives its rule-graph features from the
  simple_18 board inside the forward pass. There is no new
  `batch["primitive_features"]` tensor, no new dataset column, and no
  new collate hook.
- Smoke command (one quick CPU forward pass):

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
    tests/test_move_graph_router.py
  ```

- Static config validation:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
    scripts/validate_training_config.py --static \
    ideas/registry/p006_move_graph_router/config.yaml
  ```

- Scout training (do **not** launch unless `CLAUDE_ALLOW_TRAINING=1` is
  set in the environment by the operator):

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/train_model.py \
    --config ideas/registry/p006_move_graph_router/config.yaml
  ```

- Falsifier sweeps (matched i193 baseline, ablations sweep): use the
  primitive pipeline driver,

  ```bash
  RUN_PRIMITIVE_TRAIN=1 RUN_PRIMITIVE_DRY_RUN=0 \
    RUN_PRIMITIVE_CONFIGS=ideas/registry/p006_move_graph_router/config.yaml \
    ./run_primitive_pipeline.sh
  ```

  Each ablation (`none`, `random_edges`, `dense_edges`, `zero_delta`,
  `disable_gate`, `trunk_only`) should be run as a separate config
  with `model.ablation` overridden.
- Post-run reporting: produce slice reports via
  `scripts/reports/report_prediction_slices.py --run-dir results/<dir>
  --splits val test` and update
  `ideas/registry/p006_move_graph_router/report_template.md`.
