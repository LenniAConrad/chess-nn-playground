# Trainer Notes — a016 BT4 Primitive Mixer (legal_edge_compile_scatter)

- Use `train.py` as-is. It calls `idea_train_cli(__file__)`, which runs
  the shared idea guard before invoking the trainer.
- The guard requires `implementation_status: implemented` or `tested`,
  `implementation_kind: bespoke_model`, `device: nvidia`, and a
  registered `model.name = bt4_legal_edge_compile_scatter_mixer`.
- Static config validation (CPU-safe):

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python \
    scripts/validate_training_config.py --static \
    ideas/registry/a016_bt4_legal_edge_compile_scatter_mixer/config.yaml
  ```

- Focused test set (CPU-safe):

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest \
    tests/test_idea_registry.py tests/test_idea_reporting.py \
    tests/test_research_architectures.py -q
  ```

- GPU training (only if `CLAUDE_ALLOW_TRAINING=1`):

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python \
    ideas/registry/a016_bt4_legal_edge_compile_scatter_mixer/train.py
  ```

- The dense `(B, 64, 64, 2C)` edge-feature tensor inside the mixer is
  the dominant memory term. The default `channels = 64`,
  `num_blocks = 4`, `batch_size = 256` config is tuned to fit on a
  single consumer GPU; raise `channels` cautiously.
- Keep the standard `Trainer.fit()` metrics/report artifacts so the
  result line is comparable against sibling `mixer: conv` and
  `mixer: attention` runs from the same BT4 tower.
