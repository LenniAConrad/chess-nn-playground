# Trainer Notes

Use the guarded idea `train.py` and the existing puzzle-binary trainer path. The config remains paper-grade, CUDA-preferred, and uses the canonical tagged CRTK split.

The model returns `output["logits"]` as a single BCE-compatible puzzle margin with shape `(B,)`. Diagnostics such as escape free energy, reachable mass, cost-field summaries, and attack/defense gap can be saved with the prediction artifacts. Fine labels remain diagnostic only; they are not inputs to the model.
