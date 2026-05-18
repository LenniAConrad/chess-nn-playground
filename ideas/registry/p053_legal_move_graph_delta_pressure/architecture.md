# Architecture

`Legal-Move-Graph Pressure-Delta Primitive` (p053, LMGDP) is an
additive, gated head on top of the i193
`ExchangeThenKingDualStreamNetwork` trunk. The model consumes the
`simple_18` `(B, 18, 8, 8)` board tensor and returns one puzzle logit
plus a per-sample diagnostics dict.

LMGDP is the **pressure-delta** companion to p009 LMGConv: both
primitives compile the same per-piece-type typed legal-move adjacency
`(B, 6, 64, 64)` from the simple_18 tensor (via the shared
`_compute_typed_legal_edges` helper exposed by p009's module), but
LMGDP routes **rule-derived per-edge pressure-delta scalar features**
through that adjacency instead of routing learned per-square token
embeddings as LMGConv does. Each per-edge feature is masked by the
adjacency and aggregated two ways: per arrival square (collapsing the
source axis) and per piece type (collapsing both square axes for a
sum / mean / max summary).

## Slug deviation note

The source markdown's preferred slug `legal_move_graph_delta` is
already taken by the existing p009 LMGConv implementation
(`ideas/registry/p009_legal_move_graph_delta`). To keep both
primitives coexisting and avoid touching legacy files, this folder
uses the disambiguated slug `legal_move_graph_delta_pressure` and
the matching registry key `legal_move_graph_delta_pressure`. The
`_pressure` suffix reflects the load-bearing pressure-delta edge
features that distinguish this primitive from p009 LMGConv.

## Forward pass

1. **i193 trunk forward**. Emits `base_logit`, the joint pool feature
   (via `trunk_joint_features`), and the i193 diagnostics dict.
2. **Typed legal-move adjacency**. `_compute_typed_legal_edges(board)`
   compiles the side-to-move per-piece-type adjacency
   `edges ∈ {0, 1}^(B, 6, 64, 64)` from the piece planes plus the
   geometric attack tables and the between-square occupancy mask.
   Targets occupied by own pieces are excluded. The tensor is built
   under `torch.no_grad()` and treated as a stop-gradient float.
3. **Per-edge pressure-delta feature maps**. For each candidate edge
   `(s, t)` of piece type `r`, the model computes eight per-edge
   scalar features (all of shape `(B, 6, 64, 64)`, masked by
   `edges`):
   - `is_capture`: target has an enemy piece.
   - `into_king_zone`: target is in the enemy king's 3x3 zone.
   - `gives_check_proxy`: post-move geometric attack mask from `t`
     overlaps the enemy king square (the mover would deliver
     check from `t` ignoring source-side blocker correction).
   - `enemy_value_at_target`: weighted material value of the
     captured enemy piece (Q=9, R=5, B=3, N=3, P=1, K=0).
   - `pre_opp_attackers_at_target`: enemy attackers on `t` before
     the move, computed from `compute_attack_relations`.
   - `pre_own_defenders_at_target`: own defenders on `t` before
     the move.
   - `mover_post_attack_value_from_t`: weighted enemy attack value
     the mover of type `r` would press from `t` after the move
     (sum over `j` of geometric-attack`[r, t, j]` times enemy
     piece value at `j`). This is the load-bearing pressure-delta
     signal: it tells the head what enemy material the moving
     piece will newly threaten from the arrival square.
   - `mover_post_defender_value_from_t`: analogous over own pieces
     (defense the mover would provide from `t`).
4. **Per-target aggregation**. The eight feature maps are stacked to
   `stacked ∈ (B, 6, 64, 64, 8)` and reduced along the source axis
   to `per_target ∈ (B, 6, 64, 9)` (where the trailing +1 column is
   the per-target arrival degree).
5. **Per-type global summary**. The same `stacked` tensor is reduced
   along both square axes to `global ∈ (B, 6, 3 * 8 + 1) =
   (B, 6, 25)` containing sum / mean / max plus an edge count per
   type.
6. **Per-type per-target tokens**. Six independent `Linear(9, D)`
   layers (or one shared linear under the `shared_target_pool`
   ablation) project the per-target tensor for each piece type to
   per-square tokens `(B, 6, 64, D)`. The piece-type axis is summed
   to `(B, 64, D)`, normalised with a single `LayerNorm`, and pooled
   to a board summary via `mean + amax` over the 64 squares
   (`(B, 2*D)`).
7. **Delta MLP**. Concatenate `[board_summary, flatten(global),
   trunk_joint]` and project through `LayerNorm + Linear + GELU +
   Dropout + Linear` to a scalar `primitive_delta_raw`.
8. **Gate**. MLP over `cat(trunk_joint, edge_count_per_type,
   total_edge_count)` to sigmoid `primitive_gate`; initial bias
   `gate_init = -2.0` so the primitive starts as a near no-op.
9. **Output**. `final_logit = base_logit + primitive_gate *
   primitive_delta_raw`.

The primitive is a strict no-op on positions with no candidate edges:
every per-edge feature map is zero, the global summary collapses to
zeros and zero degrees, and the trained gate's negative-bias
initialisation keeps the gate near zero.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full LMGDP feature stack (default). |
| `no_pressure_delta` | **Primary falsifier**. Zero out the four pressure-delta features (`pre_*` and `mover_post_*`). If matches `none`, the pressure-delta story is false and the primitive is no better than a typed edge-count head. |
| `no_capture_value` | **Falsifier**. Zero out `enemy_value_at_target` and `gives_check_proxy`. If matches `none`, the explicit captured-piece value / gives-check tagging is not load-bearing. |
| `random_typed_edges` | **Falsifier**. Replace the typed adjacency with a random mask of identical per-type density. If matches `none`, the per-piece-type chess connectivity is not load-bearing. Mirrors p009's `random_typed_edges` falsifier. |
| `shared_target_pool` | **Falsifier**. Collapse the six per-piece-type per-target projections to a single shared linear. If matches `none`, the per-type routing is not load-bearing. |
| `zero_delta` | Zero primitive delta. Recovers the i193 baseline. |
| `trunk_only` | Same as `zero_delta` (semantic alias). |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder. |
| Trunk joint refeat | One additional encoder pass (the `trunk_joint_features` helper). |
| Typed-edge compile | One `(B, 6, 64, 64)` mask construction (shared with p009 LMGConv). |
| Pressure-delta features | One `(B, 6, 64, 64, 8)` per-edge feature stack and eight einsum / index reductions over the small attack tables. |
| Per-target aggregation | One sum along the source axis. |
| Per-type global summary | Three reductions (sum / mean / max) and an edge count. |
| Per-type per-target projection | Six `Linear(F+1, D)` per piece type. |
| Delta / gate MLPs | Small two-layer MLPs (~30k parameters at `head_hidden_dim = 64`). |

No per-edge trunk re-run is performed, so the wall-clock overhead is
small (low single-digit percent over i193 at B=256). The peak working
memory is the `(B, 6, 64, 64, 8)` per-edge stack which is ~786k
elements per sample.

## Implementation Binding

- Registered model name: `legal_move_graph_delta_pressure`.
- Source implementation: `src/chess_nn_playground/models/primitives/legal_move_graph_delta_pressure.py`.
- Shared helpers:
  - `legal_move_graph_delta._compute_typed_legal_edges` (typed legal-move adjacency, shared with p009)
  - `rule_graph_features.compute_attack_relations` (square-to-square attacks with occlusion)
  - `rule_graph_features.rule_geometry` (precomputed geometric attack / between / ray tables)
  - `trunk_features.trunk_joint_features` (i193 joint pool reuse)
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p053_legal_move_graph_delta_pressure/model.py`.
- Training config: `ideas/registry/p053_legal_move_graph_delta_pressure/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `MODEL_SPECS['legal_move_graph_delta_pressure']`.
