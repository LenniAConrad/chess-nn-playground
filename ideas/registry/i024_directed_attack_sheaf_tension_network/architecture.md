# Architecture

`Directed Attack-Sheaf Tension Network` implements a board-only directed sheaf model for puzzle-binary classification. It receives the repository board tensor contract `(B, C, 8, 8)` and returns one puzzle logit for the configured BCE trainer, plus diagnostic tensors for tension analysis.

## Implementation Binding

- Registered model name: `directed_attack_sheaf_tension_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/directed_attack_sheaf.py`
- Idea-local wrapper: `ideas/registry/i024_directed_attack_sheaf_tension_network/model.py`

## Components

- `EncodingAdapter`: decodes current piece occupancy, piece type, piece color, side to move, and side-relative role from `simple_18` or current-piece LC0-style planes. It does not consume labels, source tags, verification fields, or engine information.
- `DirectedAttackGraphBuilder`: builds a directed pseudo-legal attack graph from occupied source pieces. Pawn, knight, king, rook ray, bishop ray, and queen ray relations are included. Sliding pieces emit directed ray edges to squares along each line; blocked continuation edges are retained as x-ray relations with path-clear and blocked-path features.
- `SquareStateEncoder`: combines raw square planes, decoded piece/color/role one-hot features, square coordinates, and side to move into a learned square state.
- `DirectedAttackSheafLayer`: learns relation-specific source and target restriction maps \(A_\kappa, B_\kappa\), computes gated sheaf residuals \(A_\kappa z_u - B_\kappa z_v\), scatters outgoing and incoming Laplacian gradients separately, and updates square states with a directed residual block.
- `DirectedTensionReadout`: pools final square states and per-layer sheaf energy statistics. The classifier receives node-state pools, one-way versus reciprocal tension, attack and defense tension, x-ray tension, king-zone tension, gate means, path-clear means, and edge-density features.

## Forward Contract

The model accepts only a board tensor:

```text
output = model(x)
x.shape == (batch, input_channels, 8, 8)
output["logits"].shape == (batch,)
```

The diagnostic output includes `sheaf_tension`, `directed_asymmetry`, `outgoing_tension`, `incoming_tension`, `one_way_tension`, `reciprocal_tension`, `xray_tension`, `king_zone_tension`, `attack_energy`, `defense_energy`, `gate_mean`, and `edge_density`.

## Relation Geometry

Edges are typed by side-relative source role, source piece, target bucket, movement geometry, direction, and distance. Reciprocal directed edges are detected after graph construction so the readout can compare asymmetric one-way pressure against mutually constrained relations. The sheaf layer keeps source and destination restrictions distinct, so reversing an edge is not assumed to be equivalent to the original edge.
