# Architecture

`Threat-Topology Betti Bottleneck Network` is implemented as a bespoke PyTorch model with a deterministic topology bottleneck over rule-only pressure fields.

## Implementation Binding

- Registered model name: `threat_topology_betti_bottleneck_network`
- Source implementation: `src/chess_nn_playground/models/threat_topology_betti.py`
- Idea-local wrapper: `ideas/i054_threat_topology_betti_bottleneck_network/model.py`

## Modules

- `Simple18PiecePlaneAdapter` decodes `(B,18,8,8)` `simple_18` tensors into `(B,2,6,8,8)` piece planes, side-to-move, occupancy, and king maps. Unknown deterministic rule encodings fail closed.
- `RulePressureFields` builds weighted pseudo-legal attack pressure, material fields, king kernels, and the four side-relative scalar pressure fields from the math thesis.
- `RankCubicalBettiEncoder` builds rank top-k masks for `rank_ks`, then computes `beta0`, `beta1`, boundary edges, and top-k mean pressure for each field and rank.
- `ThreatTopologyBranch` maps the flattened topology tensor `(B,4,len(rank_ks),4)` into a 64-dimensional topology embedding.
- `MatchedBoardCnnStem` is a small three-layer CNN over the original board tensor, globally average- and max-pooled.
- `ThreatTopologyBettiNet` concatenates the CNN and topology embeddings and produces internal two-class logits.

## Forward Contract

Input:

```text
x: (B, 18, 8, 8)
```

Rule branch:

```text
pieces:            (B, 2, 6, 8, 8)
attack_pressure:   (B, 2, 8, 8)
material_fields:   (B, 2, 8, 8)
pressure_fields:   (B, 4, 8, 8)
topk_masks:        (B, 4, len(rank_ks), 8, 8)
topology_features: (B, 4, len(rank_ks), 4)
```

Trainer output:

```text
output["logits"]: (B,)
```

The repo's current puzzle-binary trainer uses `num_classes: 1` and BCE-with-logits. The model therefore computes internal `(B,2)` `two_class_logits` and returns their positive-minus-negative margin as `output["logits"]`. The two-class tensor is exposed as `output["two_class_logits"]` for diagnostics.

## Ablations

The implementation supports these packet-aligned ablations through `topology_ablation` or `ablation`:

- `rank_shuffle`: applies a deterministic square-rank permutation before Betti encoding, preserving each field's scalar values while destroying board adjacency.
- `histogram_only`: replaces Betti and boundary topology with non-topological top-k summary statistics.
- `no_topology_fusion`: keeps the topology pipeline active but zeroes the topology embedding before fusion.
- `degree_class_square_permutation`: permutes squares within corner, edge, and interior classes before cubical counts.
- `beta0_only` and `beta1_boundary_only`: isolate connected-component versus hole/boundary information.
- `all_one_attack_weights` and `no_target_value_bonus`: ablate pressure weighting and material target bonuses.

The default path returns diagnostics including `topology_pressure`, `betti0_mean`, `betti1_mean`, `boundary_edge_mean`, `topk_pressure_mean`, `pressure_surplus_energy`, `king_ring_pressure`, `mechanism_energy`, and `defense_gap`.
