# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = learned_relation_confidence`
- `model.token_dim`, `model.low_rank_dim`, `model.edge_hidden`,
  `model.head_hidden_dim`, `model.head_dropout`,
  `model.confidence_temperature`, `model.confidence_bias_init`,
  `model.gate_init`, `model.ablation` for the LRC head.
- trunk hyperparameters retain their i193 names with a `trunk_` prefix.
- `training.batch_size` stays at the i193 default 256; the dense
  `(B, R, 64, 64)` edge tensor fits at this batch size with
  `low_rank_dim = 8`.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (`token_dim=32, low_rank_dim=8, edge_hidden=32, B=256`)
the per-step overhead is dominated by the dense edge einsum and the
deterministic relation builder. Throughput should be within +10% of
the i193 baseline for the head itself; the relation builder's
per-relation Python loop in the pin path is the larger overhead and
is shared with i018.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: binary_only
```

Additional ablations:

- `model.ablation: scrambled_mask`  -- topology falsifier
- `model.ablation: shuffle_pieces`  -- feature falsifier
- `model.ablation: gate_only`       -- per-relation vs per-edge
- `model.ablation: no_low_rank`     -- low-rank ablation
- `model.ablation: no_edge_mlp`     -- edge MLP ablation
- `model.ablation: zero_delta`      -- i193 baseline
- `model.ablation: trunk_only`      -- strongest control
- `model.ablation: disable_gate`    -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns `lrc_mean_conf_<rel>` and `lrc_kept_<rel>`
should vary across positions; if they collapse to a per-relation
constant on the test split, the head has not learned edge-level
structure and the `gate_only` ablation should match the unablated
run.
