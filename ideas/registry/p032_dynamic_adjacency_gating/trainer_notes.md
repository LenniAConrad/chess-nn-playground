# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = dynamic_adjacency_gating`
- `model.feature_dim`, `model.head_hidden_dim`, `model.head_dropout`,
  `model.gate_init`, `model.ablation` for the DAG head
- trunk hyperparameters retain their i193 names with a `trunk_` prefix

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

8 per-type linear projections + one batched einsum on a (B, T, 64, 64)
tensor. At ``feature_dim=24`` the head wall-clock should stay within
+15% of the i193 baseline at ``B=128``. If throughput drops by more than
25%, drop one of the rarely-active move-type slots (e.g. pawn-capture)
before further scouts.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: single_move_type
```

Additional ablations:

- `model.ablation: soft_mask`        -- mask hardness
- `model.ablation: uniform_adjacency`-- rule-graph removal
- `model.ablation: shuffle_adjacency`-- rule-feature falsifier
- `model.ablation: zero_delta`       -- i193 baseline
- `model.ablation: trunk_only`       -- strongest control
- `model.ablation: disable_gate`     -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns ``primitive_gate``, ``primitive_delta``,
``dag_total_degree`` and the per-type degree diagnostics should be
inspected to confirm the gate fires preferentially on positions where a
single move-type class dominates.
