# Architecture

`King Escape Percolation Network` implements the packet's current-board, frozen-geometry king-cage operator for the `puzzle_binary` contract.

## Implementation Binding

- Registered model name: `king_escape_percolation_network`
- Source implementation file: `src/chess_nn_playground/models/king_escape_percolation.py`
- Idea-local wrapper: `ideas/registry/i051_king_escape_percolation_network/model.py`

## Input Contract

The implemented geometry adapter accepts only the repo's `simple_18` board tensor with shape `(B, 18, 8, 8)`. It decodes:

- white piece planes `0..5` and black piece planes `6..11` in `{pawn, knight, bishop, rook, queen, king}` order;
- side-to-move from channel `12`, with `1` meaning white to move and `0` meaning black to move;
- castling and en-passant planes are left available to the learned stem but are not used by the rule path-cost geometry.

Unknown encodings fail closed with a `ValueError` rather than silently applying the rule operator to unmapped channels.

## Rule Geometry

The model computes pseudo-legal attack maps from current-board occupancy only. Pawns use color-aware diagonal attacks, knights and kings use fixed leaper offsets, and bishops, rooks, and queens scan rays until the first occupied blocker. The attack maps are frozen-board hazard geometry; they do not generate legal moves, filter for king safety, call an engine, or inspect future move trees.

For each defender color, the cost-field module builds side-relative geometric planes containing defender pieces, attacker pieces, occupancy, attacker attack counts and type counts, defender defense count, normalized Chebyshev distance from the defender king, normalized distance to the board edge, side-to-move role bits, and king masks. A shared `1x1` MLP maps these features to a nonnegative learned cost. The final cell cost is:

```text
softplus(raw_cost) + base_cost + occupancy_barrier * occupied_without_defender_king
```

The defender king square is exempted from the deterministic occupancy barrier, and costs are clipped by `cost_max` for stable soft dynamic programming.

## Escape Percolation Block

For each side and each configured temperature, the model seeds a value map at that side's king and runs the packet's king-neighborhood softmin recurrence with self-loops:

```text
D_0(v) = 0 if v is the defender king square, else large_value
D_{t+1}(v) = c_s(v) - tau * logsumexp_{u in N_K(v)}(-D_t(u) / tau)
```

Snapshots are saved at `dp_snapshots`. Each saved distance map is converted into a bounded reachability-style escape map for convolutional fusion. The vector bottleneck includes edge free energies, outer-ring free energies for king-centered Chebyshev rings, reachable masses, and side-to-move aligned escape gaps/asymmetry.

The implementation also supports the packet's main cost-field controls through `ablation_mode`:

- `none`: full rule geometry and DP;
- `ring_bin_cost_shuffle`: cyclically permutes learned cost values within king-centered ring, occupancy, and coarse attacker-hazard bins before the DP;
- `no_attack_cost`: removes attack-map features from the learned cost MLP;
- `no_occupancy_barrier`: removes the deterministic occupancy barrier.

## Learned Fusion

The learned part is deliberately small:

- a one-convolution board stem with normalization, `SiLU`, and a depthwise-separable residual block;
- a shallow fusion block over `concat(stem, escape_maps)`;
- global mean and max pooling of the fused map;
- concatenation with the escape vector bottleneck;
- an MLP classifier.

The repo config keeps `num_classes: 1` for the shared binary BCE trainer, so the model computes internal two-class scores and returns the puzzle margin as `output["logits"]` with shape `(B,)`. The internal two-class scores are also returned as `two_class_logits` diagnostics.

## Outputs

Forward returns a dictionary with:

- `logits`: puzzle-binary logits, shape `(B,)` for the configured BCE contract;
- `two_class_logits`: internal non-puzzle/puzzle scores, shape `(B, 2)`;
- percolation diagnostics including `mechanism_energy`, `topology_pressure`, `king_ring_pressure`, `escape_edge_energy`, `escape_reachable_mass`, `escape_asymmetry`, `defense_gap`, `cost_field_mean`, and `cost_field_max`;
- when `return_aux=True`, decoded pieces, king masks, occupancy, attack maps, cost fields, escape maps, and escape vectors.
