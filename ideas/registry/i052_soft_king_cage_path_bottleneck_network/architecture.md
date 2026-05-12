# Architecture

`Soft King-Cage Path Bottleneck Network` implements the source packet's king-centered soft shortest-path bottleneck over a monotone, rule-derived barrier field.

## Implementation Binding

- Registered model name: `soft_king_cage_path_bottleneck_network`
- Source implementation file: `src/chess_nn_playground/models/soft_king_cage_path.py`
- Idea-local wrapper: `ideas/registry/i052_soft_king_cage_path_bottleneck_network/model.py`

## Input Contract

The implemented deterministic geometry supports the repo's `simple_18` board tensor with shape `(B, 18, 8, 8)`. It decodes piece planes `0..5` as white pieces, `6..11` as black pieces, and channel `12` as side-to-move. Castling and en-passant planes remain available to the learned trunk but are not interpreted by the rule geometry. Unknown encodings fail closed with a clear `ValueError`.

## Rule Geometry

`RuleGeometryBuilder` computes frozen-board pseudo-legal attack pressure from the current position only. Pawns use color-aware diagonals; knights and kings use fixed leaper offsets; bishops, rooks, and queens scan rays through empty squares and stop after the first occupied blocker. The model does not generate legal moves, count legal moves, inspect king safety after moves, call an engine, or search a game tree.

For each defender king, the geometry provides own occupancy, opponent occupancy, opponent attack pressure, own defense pressure, Chebyshev distance from the king, edge distance, coordinate features, and side-to-move role bits.

## Monotone Barrier Field

`MonotoneBarrierField` builds nonnegative per-side barrier maps:

```text
b_c(i) = softplus(base)
       + softplus(w_attack) * log1p(attacks_by_opponent)
       + softplus(w_own) * own_occupancy
       + softplus(w_opp) * opponent_occupancy
       + softplus(local_1x1_adapter(features))
```

The positive coefficients make attacked and occupied squares harder, not easier, to traverse. The local adapter calibrates king distance, edge distance, side-to-move, king maps, and piece-count context while keeping the final barrier nonnegative and clipped for numerical stability.

## Soft Escape DP

`SoftKingEscapeDP` runs the packet's absorbing-target Bellman-Ford recurrence. For each defender color, shell radius, and temperature, target squares are all squares whose Chebyshev distance from the king is at least that radius:

```text
T_r(k) = { i : d_infty(i, k) >= r }
```

The value map is initialized to zero on the target shell and `dp_big_m` elsewhere. Each step applies:

```text
V_{t+1}(i) = 0 if i in T_r(k)
V_{t+1}(i) = softmin_tau({ V_t(j) + b_c(j) : j in N_8(i) }) otherwise
```

The cage scalar is the final value sampled at the defender king. The model exports final distance fields `(B, 2, R, Q, 8, 8)`, cage scalars `(B, 2, R, Q)`, side-to-move-relative cage gaps, and a temperature-spread path-entropy proxy.

The central topology ablation is implemented as `ablation_mode: random_grid_degree_preserving`, which replaces the true 8-neighbor board table with a fixed deterministic random neighbor table that preserves each square's outgoing degree. `shell_shuffled_barrier` cyclically shuffles barrier values inside king-centered Chebyshev shells before the DP.

## Fusion Head

The learned branch is a small CNN trunk with residual blocks. If `use_distance_fields` is enabled, bounded transforms of the final DP fields are projected by a `1x1` convolution and concatenated with the trunk feature map. The head globally mean/max pools the fused map, appends the low-dimensional cage feature vector, and produces internal two-class scores.

The repo's current i052 config uses `num_classes: 1` for the shared BCE trainer, so the model returns the puzzle margin as `output["logits"]` with shape `(B,)` and exposes the internal `(B, 2)` scores as `two_class_logits`.

## Outputs

Forward returns a dictionary containing:

- `logits`: BCE-compatible puzzle logits, shape `(B,)`;
- `two_class_logits`: internal non-puzzle/puzzle scores, shape `(B, 2)`;
- diagnostics including `cage_energy`, `side_to_move_cage_gap`, `topology_pressure`, `king_ring_pressure`, `path_entropy_proxy`, `cage_asymmetry`, `barrier_mean`, `barrier_max`, `attack_barrier_weight`, `occupancy_barrier_weight`, `defense_gap`, and `trunk_feature_energy`;
- when `return_aux=True`, decoded pieces, king maps, attack pressure, barrier fields, distance fields, cage scalars, cage features, and target-shell masses.
