# Trainer Notes

Use the guarded idea `train.py` and the existing puzzle-binary trainer. The model is board-only and consumes no CRTK/source metadata as neural input.

The repo config uses:

```text
mode: puzzle_binary
model.num_classes: 1
training.loss: bce_with_logits
```

So `ThreatTopologyBettiNet.forward` returns `output["logits"]` with shape `(B,)`. The internal two-class logits are retained as `output["two_class_logits"]` for diagnostics and architecture inspection.

For fair ablations, change only `model.topology_ablation` or `model.ablation` and keep the split, class weighting, optimizer family, batch size policy, and reporting pipeline fixed.
