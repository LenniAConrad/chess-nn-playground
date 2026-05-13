# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config mirrors
the i193 baseline (same split, encoding, seed, budget, threshold-rule).

Differences vs i193:

- `model.name = sparse_legal_graph_transition`
- `model.feature_dim`, `model.edge_hidden_dim`, `model.head_hidden_dim`,
  `model.head_dropout`, `model.gate_init`, `model.ablation` for the
  SLMGT head
- trunk hyperparameters retain their i193 names with a `trunk_` prefix
- ``training.batch_size`` reduced to 64 to leave headroom for the
  (B, 64, 64, edge_hidden) pair tensor.

## Loss

`bce_with_logits` on the puzzle logit.

## Cost expectation

At defaults (``feature_dim=16``, ``edge_hidden_dim=24``, B=64) the
pair tensor uses ~25MB. The dominant per-step cost is the Linear
projections + LayerNorm over the pair tensor; throughput should be
within +20% of the i193 baseline. If throughput drops by more than
30%, halve ``edge_hidden_dim`` before further scouts.

## Ablation runs

Primary falsifier:

```yaml
model:
  ablation: separable_phi
```

Additional ablations:

- `model.ablation: uniform_adjacency` -- rule-graph-removal control
- `model.ablation: shuffle_adjacency` -- rule-feature falsifier
- `model.ablation: zero_delta`        -- i193 baseline
- `model.ablation: trunk_only`        -- strongest control
- `model.ablation: disable_gate`      -- gate load-bearing

## Reports

Standard idea report; required slices listed in `report_template.md`.
The diagnostic columns ``primitive_gate``, ``primitive_delta``,
``slmgt_degree_mean``, ``slmgt_edge_norm``, and ``slmgt_edge_max``
should be inspected to confirm the gate fires preferentially on
tactically active positions.
