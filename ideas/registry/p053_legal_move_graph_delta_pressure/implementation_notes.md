# Implementation Notes

## Slug / folder choice

The source primitive markdown prefers the slug `legal_move_graph_delta`
and the folder `p053_legal_move_graph_delta`. That slug is already
used by the existing p009 implementation
(`ideas/registry/p009_legal_move_graph_delta`) which registers the
model name `legal_move_graph_delta` in
`src/chess_nn_playground/models/_registry_manifest.py`. The idea
registry validator enforces `config.yaml model.name == idea.yaml
slug == folder slug suffix`, and the model registry is a global
dictionary, so the two implementations cannot share that name.

Renaming p009 would break its tests
(`tests/test_legal_move_graph_delta.py`), INDEX entry, TODO entry,
and any external references to its registry key. The repo's global
rule "Do not move or delete legacy files unless this task explicitly
requires it" applies. So the new primitive uses the disambiguated
slug `legal_move_graph_delta_pressure` and the matching folder
`p053_legal_move_graph_delta_pressure`. The `_pressure` suffix is
descriptive: it reflects the load-bearing per-edge pressure-delta
features that distinguish this primitive from p009 LMGConv. The
idea_id is still `p053` per the task spec, and the source markdown
is referenced in this folder's docs unchanged.

## Module layout

- `legal_move_graph_delta_pressure.LegalMoveGraphDeltaPressure` --
  the model class.
- `legal_move_graph_delta_pressure.build_legal_move_graph_delta_pressure_from_config`
  -- canonical builder; the registry manifest points at this.
- `legal_move_graph_delta_pressure.compute_pressure_delta_edge_features`
  -- pure function that, given a board, a typed-edge mask and the
  geometry tables, returns the eight per-edge scalar feature maps.
- `legal_move_graph_delta_pressure.aggregate_per_target_features` --
  pure function: per-target sum of stacked features (with the
  arrival degree column appended).
- `legal_move_graph_delta_pressure.per_type_global_summary` --
  pure function: per-(b, r) sum + mean + max + edge count.

The pure functions are exposed for unit-testing the rule-derived
feature paths in isolation from the trunk forward.

## Reuse of p009 topology

The typed legal-move adjacency `(B, 6, 64, 64)` is reused directly
from p009's `_compute_typed_legal_edges` helper. This is intentional:
both primitives should agree on what counts as a candidate move so
the only difference is the **routing semantics**. If p009's
topology helper changes (e.g. to add castling / en-passant edges),
both primitives inherit the change automatically.

## Pressure-delta proxy

The `mover_post_attack_value_from_t` and
`mover_post_defender_value_from_t` features use the **unoccluded
geometric** attack table `geom_attacks[r, t, j]` instead of an
explicitly recomputed occluded one. The proxy:

```
post_attack_value(b, r, t) = sum_j geom_attacks[r, t, j] * V_enemy[b, j]
```

ignores the change in blockers caused by the source piece moving. For
non-sliders (P, N, K) this is exact. For sliders (B, R, Q) it is a
conservative over-estimate of post-move attacks (the moving piece may
also be a blocker of a stronger ray it now extends). In practice this
adds a small bias toward sliders but is dramatically cheaper than
re-running `compute_attack_relations` per-edge. A future iteration may
correct this by subtracting the source piece's blocker contribution on
the post-move ray; that correction is listed as an out-of-scope
ablation in `ablations.md`.

## Numerics and AMP

- Topology compile: under `@torch.no_grad()` and a float mask. The
  cumulative blocker scan and target-open mask are computed on
  small `(B, 8, 8)` planes so no autocast issues arise.
- Per-edge feature stack: the largest tensor is the
  `(B, 6, 64, 64, 8)` per-edge stack. At `B = 256` this is ~25 MB
  fp32. Acceptable for the trainer's default AMP regime.
- `index_add_` / scatter ops: not used here -- the model uses
  `einsum` and broadcasting plus `sum` / `mean` / `amax` reductions
  which keep the autograd graph simple and AMP-safe.
- Per-target / global aggregation: sum and mean over square axes
  use fp32 promotion of summed values when AMP forwards push the
  encoder to half precision. The aggregation MLPs run after a
  LayerNorm which restores numerical scale.
- The final loss uses `BCEWithLogitsLoss` -- never insert an
  explicit sigmoid before the criterion under AMP (PyTorch AMP
  notes flag `binary_cross_entropy` / `BCELoss` as unsafe in
  autocast-enabled regions; we keep the single-logit interface and
  never insert an explicit sigmoid).

## Stop-gradient discipline

The typed-edge adjacency tensor is constructed under
`torch.no_grad()` (matching p009 LMGConv). The per-edge feature
maps inherit this stop-gradient property because all of their
inputs (piece planes, geometric attack tables, pre-move attacker
counts via `compute_attack_relations`) are themselves functions of
the input board and constant tables, neither of which has a
gradient with respect to model parameters. Only the per-type
per-target projection, the delta MLP, and the gate MLP carry
gradients.

## Reporting

Diagnostics emitted by `forward`:

- `lmgdp_total_edge_count` -- total candidate edges (B,).
- `lmgdp_edge_count_{P, N, B, R, Q, K}` -- per-type edge counts.
- `lmgdp_post_attack_value_mean_{...}` -- per-type mean
  `mover_post_attack_value_from_t` across candidate edges.
- `lmgdp_capture_value_mean_{...}` -- per-type mean
  `enemy_value_at_target` across candidate edges.
- All trunk diagnostics under the `trunk_<name>` prefix.
- Standard primitive head diagnostics (`base_logit`,
  `primitive_delta`, `primitive_gate`, ...).

## Production / precompute

Precomputing the eight per-edge feature maps into Parquet columns
is *out of scope* for this initial implementation. The features
depend on the full per-batch candidate adjacency `(B, 6, 64, 64)`
which is ~25 MB per sample in float32; columnar storage at row
granularity would mean storing the per-target aggregation `(B, 6,
64, 9)` instead, which is small enough to be feasible (~14 KB per
sample). That precompute path is left as a follow-up if and only if
the in-forward path proves too slow at the trainer's intended
throughput.
