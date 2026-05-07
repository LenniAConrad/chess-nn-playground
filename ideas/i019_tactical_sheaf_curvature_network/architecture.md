# Architecture

`Tactical Sheaf Curvature Network` (`TSCN`) realizes the source packet's typed
sheaf-frustration / target-centered curvature operator as a bespoke PyTorch
model for the repo's `puzzle_binary` task.

## Implementation Binding

- Registered model name: `tactical_sheaf_curvature_network`
- Source implementation file: `src/chess_nn_playground/models/tactical_sheaf_curvature.py`
- Idea-local wrapper: `ideas/i019_tactical_sheaf_curvature_network/model.py`

## Modules

`BoardChannelAdapter` consumes a `(B, C, 8, 8)` board tensor, applies a
configurable depth of `1x1` plus `3x3` conv stages with GELU and dropout, and
emits per-square node features of shape `(B, 64, d_node)`. It is encoding-
agnostic: any `C` is supported through the leading `1x1` conv, so `simple_18`,
`lc0_static_112`, and `lc0_bt4_112` all flow through the same trunk.

`TypedRelationComplex` precomputes a fixed directed candidate-relation list
over the 64 squares from chess geometry alone. The packet's `include_*` knobs
toggle which relation families participate. With the default flags the complex
contains:

- 8 ray directions (north, south, horizontal east/west, four diagonals) at
  distances 1 to 7, optionally tied across the file mirror so east and west
  rays share parameters.
- Knight jumps with a forward / backward type split.
- King-neighborhood edges.
- Oriented pawn-capture candidates (forward and backward halves kept distinct
  so pawn direction is preserved).

Each directed edge carries an integer `edge_type`, a `relation_group` ID for
group-pooled statistics, and an 8-D geometry vector (normalized source/target
ranks and files, signed deltas, distance, distance bucket). Buffers are
registered as non-persistent tensors so they ride the module to GPU without
being saved in checkpoints.

`SheafRestrictionGenerator` mixes a learned type embedding with the geometry
vector through a small MLP and emits two diagonal restrictions
`a, b in 1 + 0.5 * tanh(.)`, bounding the spectral effect of the sheaf
coboundary at initialization.

`SheafGate` is an input-dependent sparse gate: it scores each edge from
`(x_src, x_dst, q)` through a 2-layer MLP and applies `sigmoid` so each gate
lies in `(0, 1)`.

`TacticalSheafLayer` realizes one round of typed sheaf diffusion. For every
directed edge `e = (u -> v, r)` with bounded restrictions `a, b` and gate
`g_e` it computes the sheaf coboundary

```text
delta_e = b * x_v - a * x_u
edge_energy_e = g_e * ||delta_e||_2^2
```

The Laplacian-like residual update scatters `b * g_e * delta_e` onto target
squares, subtracts the symmetric source-side scatter, and divides by the
edge-weighted node degree, giving a stable diffusion step
`x <- LayerNorm(x - eta * lap_update + NodeMLP(x))` with a sigmoid-bounded
`eta in (0, 1)`. The same layer also emits target-centered curvature: for
each destination square it forms the gate-weighted variance of the
transported claims `a * x_u`, an inexpensive 2-cell proxy that reads
"how much do the incoming tactical claims on this target disagree?".

`CurvatureStatsPool` concatenates per-layer statistics with mean / max / std
node pooling. Each layer contributes 8 scalar stats (edge-energy mean / std /
max, gate mean, gate entropy, curvature mean / std / max) plus per-relation-
group means of edge energy and gate, so the readout grows linearly in
`(num_layers, group_count)`.

The classifier head is a `LayerNorm -> Linear -> GELU -> Dropout -> Linear`
MLP that produces one BCE-compatible puzzle logit when `num_classes = 1`.

## Diagnostics

The `forward` returns a dict containing:

- `logits` (BCE-compatible; shape `(B,)` when `num_classes = 1`).
- `sheaf_frustration`: pooled per-layer mean edge energy.
- `curvature_mean`, `curvature_max`: target-centered curvature averaged over
  layers and squares.
- `gate_mean`, `gate_entropy`: sparsity / saturation diagnostics on the
  learned edge gates.
- `ray_energy`, `jump_energy`, `pawn_candidate_energy`: relation-group
  averaged edge energy across layers.
- `relation_gate_pressure`: relation-group averaged gate strength.
- `node_stalk_std`: spread of the final stalk states across squares.

Diagnostics are reporting-only; they do not enter the training loss.

## Contract

- Input: `(B, C, 8, 8)` board tensor only. CRTK / verification / source / engine
  metadata is reporting-only and is not consumed.
- Output: dict with `logits` of shape `(B,)` for the one-logit puzzle_binary
  BCE-with-logits trainer, plus the diagnostics listed above.
- Symmetry: only the file-mirror tying is applied by default; pawn direction
  and side-to-move are intentionally not assumed equivariant, matching the
  source packet's symmetry stance.
