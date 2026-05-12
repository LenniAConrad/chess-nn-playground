# Trainer Notes

Use the guarded idea entrypoint:

```bash
python ideas/registry/i012_dykstra_lcp/train.py
```

The config requires `device: nvidia`, uses the canonical CRTK tagged split, and trains with `training.loss: dykstra_lcp`.

The model `forward(x)` consumes only the board tensor. The trainer may save fine-label columns in predictions for diagnostics, but fine labels are not passed to the model or loss.
