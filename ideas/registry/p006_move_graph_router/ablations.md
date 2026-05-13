# Ablations — p006 Move-Graph Router

## Switches (model.ablation)

| Mode | What it tests |
|---|---|
| `none` | Full architecture (default). |
| `random_edges` | Replace the legal-edge mask with a random mask of identical density. **This is the primary MGR falsifier**: if the architecture matches `random_edges`, the rule structure (not the sparsity) was not load-bearing and the primitive should be dropped. |
| `dense_edges` | Use a fully-connected mask. Tests whether the operator collapses to a generic 64x64 mixer when the sparse adjacency is removed (the "hidden rebrand" failure mode flagged in external_02). |
| `zero_delta` | Hold `primitive_delta = 0`. Equivalent to running the i193 trunk alone — architecture-level baseline. |
| `disable_gate` | Hold `primitive_gate = 1`. Tests whether the gate is load-bearing or whether the head's raw delta is already usable. |
| `trunk_only` | Zero out both features and delta (strict no-op head). Minimum control. |

## Falsification criteria

Promote p006 only if, under `model.ablation = none`:

- Aggregate PR AUC delta from i193 >= -0.005.
- CRTK class-1 matched-recall FP rate drops by at least 5% relative.
- Wall-clock per epoch within 1.3x of i193.

Drop p006 if any of:

- `random_edges` matches the `none` architecture (rule structure was not
  load-bearing — operator is just a sparse 64x64 mixer with random mask).
- `dense_edges` outperforms `none` (the sparsity itself is the bug, not
  the feature).
- `zero_delta` matches `none` (the delta is noise — gate and trunk
  diagnostics suffice).

## Deferred internal proposals from external_02

The source primitive packet
(`ideas/research/primitives/external_02_move_graph_router_delta_accumulator.md`)
ranks five primitive candidates. Per the implementation rule "implement
the strongest or first-ranked proposal" only **MGR** is implemented here;
the remaining four are deferred:

- **IDA — Delta-Accumulator Primitive**: stateful affine layer with
  exact ClippedReLU regime propagation. Deferred because (a) chess-NN-
  playground's training loop is dense-batch, not search-trajectory, so
  IDA's headline speed advantage is invisible to the scout, and (b)
  its core property (`O(‖Δx‖_0 · d_out)`) is an *inference-time*
  primitive whose scout-scale falsifier needs a search-trajectory
  benchmark this batch does not have.
- **KISB — King-Indexed Switching Bank**: bank of weight matrices
  indexed by king square, with a cache-refresh operator. Deferred
  because the file's own ranking warns it is "the highest-risk
  proposal on the hidden-rebrand axis" — it should be evaluated only
  after IDA shows ≥1.5x throughput edge, which we have not measured.
- **TBL — Tropical Bilinear**: (max,+) semiring bilinear form.
  Deferred because the file explicitly classifies it as "underexplored
  primitive for chess, not novel", and the file's own ranking puts it
  fourth.
- **EEP — Equilibrium Energy Primitive**: implicit-layer fixed-point
  over a learned pairwise energy. Deferred because the file's failure-
  mode catalogue lists "trains but underperforms" as the most likely
  scout-scale outcome, matching the i242 attention-is-data-hungry
  finding the playground has already paid for once.

If any of the four prove relevant after MGR's scout, they should be
promoted under fresh `p###` IDs.

## Cross-references

- Source primitive packet:
  `ideas/research/primitives/external_02_move_graph_router_delta_accumulator.md`.
- Shared rule-graph helpers:
  `src/chess_nn_playground/models/primitives/rule_graph_features.py`.
- Training plan and falsifier framing:
  `ideas/research/primitives/PRIMITIVE_TRAINING_TODO.md`.
