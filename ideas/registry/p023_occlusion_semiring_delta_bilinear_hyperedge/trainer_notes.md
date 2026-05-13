# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). Matches the i193
baseline on split, seed, encoding, training budget, and threshold-
selection rule.

Differences vs the i193 baseline:

- `model.name = occlusion_semiring_delta_bilinear_hyperedge`
- `model.token_dim`, `model.hidden_dim`, `model.bilinear_dim`,
  `model.head_hidden_dim`, `model.head_dropout`, `model.ablation`
  for the bilinear-hyperedge head.
- Trunk hyperparameters retain their i193 names with a `trunk_`
  prefix.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

The Python-side backward recurrence walks `RAY_MAX_LEN = 7`
iterations per sample. The bilinear hyperedge step adds 4 Hadamard
products per square. Throughput should be within ~15% of i193. A
fused CUDA / Triton recurrence is the production speed path
(deferred).

## Ablation runs

Promotion requires the falsifier ablation:

```yaml
model:
  ablation: disable_bilinear
```

with everything else matched. If the no-bilinear run matches the
unablated run on the declared slice, the bilinear hyperedge claim
fails and the primitive should be dropped (or the head should be
reduced to a non-bilinear ray reducer).

## Reports

Standard idea report (see `report_template.md`). Required slices:

- Aggregate validation and test PR AUC.
- Near-puzzle FP rate at matched recall.
- Slice PR AUC for through-the-square motifs (pins, skewers, x-rays,
  batteries along a single line).
- `osdb_pair_hyperedge_magnitude` distribution vs label.
- Cost: params, FLOPs/MACs, throughput.
