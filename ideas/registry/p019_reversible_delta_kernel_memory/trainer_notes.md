# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade and CUDA-required, mirroring the i193 baseline so the
architecture-level comparison is matched on:

- same train/val/test split
- same encoding (`simple_18`)
- same seed
- same training budget and early-stopping policy
- same threshold-selection rule

Differences vs the i193 baseline:

- `model.name = reversible_delta_kernel_memory`
- `model.token_dim`, `model.memory_heads`, `model.memory_value_dim`,
  `model.num_queries`, `model.head_hidden_dim`, `model.head_dropout`,
  `model.ablation` for the kernel-memory head.
- All trunk hyperparameters retain their i193 names with a `trunk_`
  prefix in the config so the builder can forward them to the wrapped
  `ExchangeThenKingDualStreamNetwork`.

## Loss

`bce_with_logits` on the puzzle logit. No primitive-specific auxiliary
loss is required -- the head learns through the main BCE signal.

## Cost expectation

The static kernel-memory forward is `O(64 * h * v)` per sample, the
same order of magnitude as a single 1x1 conv at the same width.
Training throughput should be within 10% of i193 at scout scale.

## Ablation runs

Promotion of p019 requires the falsifier ablation:

```yaml
model:
  ablation: shuffle_tokens
```

with everything else matched to the unablated run. If the shuffled-
token run matches the unablated run on the declared target slice, the
kernel memory is not load-bearing and the primitive should be dropped.

## Reports

Standard idea report (see `report_template.md`). Required slices:

- Aggregate validation and test PR AUC.
- Near-puzzle false-positive rate at matched recall.
- Slice PR AUC for the declared target slices (pin / overloaded
  defender / king-piece distance).
- `primitive_gate` mean on positive vs negative samples (to confirm
  the gate fires on positions that need it).
- Cost: params, FLOPs/MACs, throughput, wall-clock per epoch.
