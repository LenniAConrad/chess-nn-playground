# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). Mirrors the i193
baseline so the architecture-level comparison is matched on split,
seed, encoding, training budget, and threshold-selection rule.

Differences vs the i193 baseline:

- `model.name = blocker_reset_ray_scan`
- `model.token_dim`, `model.hidden_dim`, `model.head_hidden_dim`,
  `model.head_dropout`, `model.ablation` for the ray-scan head.
- Trunk hyperparameters retain their i193 names with a `trunk_`
  prefix.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

The Python-side scan loop walks `RAY_MAX_LEN + 1 = 8` iterations per
direction per sample. At scout scale this is comparable to one extra
conv layer; throughput should be within ~15% of i193. A fused CUDA /
Triton segmented-scan kernel is the production speed path (deferred).

## Ablation runs

Promotion requires the falsifier ablation:

```yaml
model:
  ablation: zero_blocker
```

with everything else matched. If the no-blocker run matches the
unablated run on the declared slice, the blocker reset is not
load-bearing and the primitive should be dropped.

## Reports

Standard idea report (see `report_template.md`). Required slices:

- Aggregate validation and test PR AUC.
- Near-puzzle false-positive rate at matched recall.
- Slice PR AUC for sliding-piece-dependent slices (pin / skewer /
  discovered attack / rook-on-open-file).
- `primitive_gate` mean on positions with at least one slider vs
  positions without.
- Cost: params, FLOPs/MACs, throughput.
