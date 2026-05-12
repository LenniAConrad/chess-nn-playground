# Architecture

`Non-Backtracking Tactical Walk Network` is implemented as a bespoke PyTorch model
that builds a directed current-board attack/protection edge graph and propagates
typed messages along its non-backtracking edge-line transitions.

## Implementation Binding

- Registered model name: `non_backtracking_tactical_walk_network`
- Source implementation: `src/chess_nn_playground/models/trunk/non_backtracking_tactical_walk.py`
- Idea-local wrapper: `ideas/registry/i055_non_backtracking_tactical_walk_network/model.py`

## Modules

- `Simple18BoardParser` decodes `(B,18,8,8)` `simple_18` tensors and fails closed for
  any other deterministic encoding. CRTK/source metadata is never read.
- `AttackProtectionEdgeBuilder` enumerates pseudo-legal attack/protection edges per
  board on CPU. For each occupied source piece it produces:
  - `enemy_piece_attack` edges to occupied enemy targets,
  - `friendly_piece_protect` edges to occupied own-color targets,
  - `enemy_king_zone_attack` edges to virtual nodes for enemy king-zone squares,
  - `own_king_zone_protect` edges to virtual nodes for own king-zone squares.
  Sliders' rays terminate on the first occupied square (reaching the blocker).
- `NonBacktrackingTransitionBuilder` connects edge `e=(u→v)` to edge `f=(v→w)` only if
  `v` is an occupied node (virtual king-zone targets have no outgoing transitions) and
  `w != u`, exactly the Hashimoto non-backtracking transition relation. Ablation modes
  `backtracking_allowed` and `randomized_transitions` are supported as central
  falsifiers; the latter permutes destination edges within `(rel(e), rel(f))` buckets,
  preserving relation-pair counts and per-relation degree marginals.
- `TypedEdgeEncoder` lifts deterministic per-edge features (relation one-hot, source
  and target piece-type one-hots, side-relative coordinates and displacement,
  distance, color-relative flags) to the edge-state width.
- `NonBacktrackingEdgeBlock` is a typed scatter-add propagation layer with a basis
  decomposition `W_shared + sum_j alpha[type_pair, j] W_j` so the per-`type_pair`
  weights stay compact, plus a per-relation bias and a residual LayerNorm.
- `EdgeMomentPooler` produces global mean, max, log-sum-exp, per-relation means, and
  an energy scalar over valid edges. Pooled summaries from layer 0 through `K` are
  concatenated into the edge latent.
- `SmallBoardAdapter` is a compact two-layer convolutional trunk over the 18 input
  planes; it is intentionally small and not the central claim.
- `NonBacktrackingTacticalWalkNet` concatenates the board latent and the edge latent,
  passes them through an MLP, and returns puzzle logits plus diagnostics.

## Forward Contract

Input:

```text
x: (B, 18, 8, 8)
```

Per-board edge construction (CPU):

```text
edge_features:        (B, edge_max, F_edge=32)
edge_mask:            (B, edge_max)
edge_relation:        (B, edge_max)
transition_src:       (B, transition_max)
transition_dst:       (B, transition_max)
transition_type_pair: (B, transition_max)
transition_mask:      (B, transition_max)
edge_overflow:        (B,)
transition_overflow:  (B,)
```

Per-block propagation:

```text
incoming_sum[dst] = scatter_add(typed_message(src_state[src], type_pair))
edge_state = LayerNorm(edge_state + GELU(self_linear(edge_state) + incoming_sum + bias[rel]))
```

Trainer output:

```text
output["logits"]: (B,)
```

The repo's puzzle-binary trainer uses `num_classes: 1` with BCE-with-logits, so the
model computes internal `(B, 2)` `two_class_logits` and exposes
`output["logits"] = two_class[:, 1] - two_class[:, 0]`. The two-class tensor is also
reported as `output["two_class_logits"]` for diagnostics.

## Diagnostics

`output` includes:

- `non_backtracking_walk_energy`, `mechanism_energy` — pooled edge-state energy
- `edge_count`, `transition_count`, `edge_overflow_count`, `transition_overflow_count`
- `enemy_attack_edge_count`, `friendly_protect_edge_count`,
  `enemy_king_zone_edge_count`, `own_king_zone_edge_count`
- `edge_state_mean_norm`, `edge_state_max_norm`, `proposal_profile_strength`,
  `proposal_keyword_count`
- `defense_gap` and `king_ring_pressure` summaries from edge-relation counts

## Ablations

- `none` (default) — full non-backtracking continuation semigroup.
- `backtracking_allowed` — drop the `w != u` non-backtracking exclusion. Tests whether
  the no-immediate-return constraint matters.
- `randomized_transitions` — central falsifier. Permutes destination edges within
  `(rel(e), rel(f))` buckets; preserves edge tokens, edge counts, relation-pair
  counts, and per-relation degree marginals while destroying the actual continuation
  relation. If the main model does not beat this control, abandon the mechanism.
