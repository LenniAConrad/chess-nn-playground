# Trainer Notes

Use the guarded idea `train.py` and the shared `puzzle_binary` training path. The model returns `output["logits"]` as a single BCE-compatible puzzle margin with shape `(B,)`.

The cage diagnostics are safe prediction artifacts: cage energy, side-to-move cage gap, path-entropy proxy, barrier summaries, and attack/occupancy coefficients. Fine labels remain diagnostic only and must not be passed as model inputs.
