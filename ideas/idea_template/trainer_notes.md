# Trainer Notes

- Use `train.py` as-is. It calls `idea_train_cli(__file__)`, which checks idea/config/model identity before training.
- The guard requires `implementation_status` to be `implemented` or `tested`, `device: nvidia`, and a registered `model.name`.
- Promote the draft `config.yaml` into a full shared-trainer config before training: include `run`, top-level `mode`, `data`, `model`, and `training`.
- Keep special losses inside the model/trainer extension only when the shared benchmark loss is insufficient; document any deviation here.
- Keep the standard metrics/report artifacts from `Trainer.fit()` so idea runs remain comparable.
