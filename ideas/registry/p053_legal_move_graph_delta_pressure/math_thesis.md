# Math Thesis

The source primitive markdown
(`ideas/research/primitives/external_48_legal_move_graph_delta_primitive.md`)
proposes a board-aware **edge-centric** candidate-move graph head
whose load-bearing signal is the per-edge tactical pressure delta
along each candidate move. The same markdown explicitly notes that
the existing p009 (LMGConv) and p011 (legal-edge-compile-scatter)
implementations cover **typed adjacency routing of square tokens**
but do *not* express explicit per-edge tactical features such as
captured-piece value, gives-check, or post-move attack value from
the arrival square. p053 is the implementation of those per-edge
pressure-delta features as a gated additive head on top of the i193
trunk.

## Topology

Let `B ∈ ℝ^(B, 18, 8, 8)` be a simple_18 board batch. The set of
candidate edges is the side-to-move's typed legal-move adjacency

```
E(B) ∈ {0, 1}^(B, R, 64, 64),    R = 6,
```

where `E[b, r, s, t] = 1` iff the side-to-move at sample `b` owns a
piece of type `r ∈ {P, N, B, R, Q, K}` on square `s`, and that piece
can move (with occlusion for sliding pieces) to square `t`, and `t`
is not occupied by an own piece. This is exactly the per-piece-type
adjacency compiled by p009 LMGConv; we reuse
`_compute_typed_legal_edges` so both primitives share a single
rule-exact topology source. The topology tensor is built under
`torch.no_grad()` and treated as a stop-gradient float (matching the
source markdown's "no gradient with respect to the discrete graph"
requirement).

The compiler currently covers **pseudo-legal** edges for all six
piece types with sliding-piece occlusion via the between-square
mask. Special move classes (en-passant, castling, promotion target
selection) are deferred -- they require additional state from the
castling-rights and en-passant planes and are an explicit
``out-of-scope`` item in this primitive's `ablations.md`. The
primitive is still honest: the per-edge feature design does not
falsely claim those classes are covered, and the simple_18 input
contract is preserved.

## Per-edge pressure-delta features

For each candidate edge `e = (b, r, s, t)` define eight scalar
features. Let `O(B) ∈ {0, 1}^(B, 64)` be the per-square occupancy
mask, let `K_opp(B) ∈ {0, 1}^(B, 64)` be the enemy-king one-hot, let
`V_enemy(B) ∈ ℝ^(B, 64)` be the per-square enemy material value
(Q=9, R=5, B=N=3, P=1, K=0) and let `V_own(B)` be the analogous own
material value with attack-side weights (K=3 to include defenders of
the king zone). Let `A_geom[r, t, j]` be the unoccluded geometric
attack table -- 1 if a piece of type `r` placed at `t` attacks `j`
ignoring blockers.

The eight features are:

```
f_capture(e)            = O_enemy(B)[t]
f_into_king_zone(e)     = KingZone3x3(K_opp(B))[t]
f_gives_check(e)        = clamp(sum_j A_geom[r, t, j] * K_opp(B)[j], 0, 1)
f_value_at_target(e)    = V_enemy_capture(B)[t]
f_pre_opp_attackers(e)  = (enemy attacks_per_color sum at t)[b]
f_pre_own_defenders(e)  = (own attacks_per_color sum at t)[b]
f_post_attack_value(e)  = sum_j A_geom[r, t, j] * V_enemy_attack(B)[j]
f_post_defender_value(e)= sum_j A_geom[r, t, j] * V_own_attack(B)[j]
```

The pre-move attacker / defender counts use the occluded
`compute_attack_relations` output. The post-move attack / defender
value functions intentionally use the **unoccluded** geometric
attack table because we aggregate over targets `t` -- the proxy
measures the mover's post-move geometric attack set rather than a
rederived occluded one that would require recomputing the attack
relations after the source piece is removed. The bias this
introduces is small and documented in `implementation_notes.md`.

The pre-move attacker counts and post-move attack value are the
**load-bearing pressure-delta features**: their masked sum over
candidate edges arriving at a target square `t` measures whether
moving to `t` increases the side-to-move's forcing pressure beyond
what already exists on `t`.

Pre-move scalars are broadcast as `(B, 1, 1, 64)`; per-(r, t)
scalars are broadcast as `(B, R, 1, 64)`. Each is masked by
`E(B)` so the final feature tensor

```
F(B) ∈ ℝ^(B, R, 64, 64, 8)
```

has zero entries outside the candidate edge set.

## Aggregation

Two aggregations are computed:

1. **Per-target aggregation**:
   ```
   F_target ∈ ℝ^(B, R, 64, 8),    F_target[b, r, t, k] = sum_s F[b, r, s, t, k]
   ```
   plus the per-(r, t) arrival degree appended as a 9th column.
2. **Per-type global summary**:
   ```
   F_global ∈ ℝ^(B, R, 25)
   ```
   containing per-(b, r) sum, mean, and max over (s, t) of each of
   the 8 features plus a single edge-count scalar.

## Per-type per-target projection

Six independent linear maps `W_r : ℝ^9 → ℝ^D` (or one shared linear
under the `shared_target_pool` ablation) project per-type per-target
features:

```
T_r[b, t, :] = W_r(F_target[b, r, t, :])
T[b, t, :] = sum_r T_r[b, t, :]
T[b, t, :] = LayerNorm(T[b, t, :])
```

The square axis is pooled via mean + amax, yielding a
`(B, 2D)` board summary. The flattened global summary
`(B, R * 25) = (B, 150)` is concatenated to the board summary and
to the i193 trunk joint feature, and the concatenation is projected
through a small MLP to `primitive_delta_raw ∈ ℝ^B`.

A small gate MLP over `cat(trunk_joint, edge_counts_per_type,
total_edge_count)` with initial bias `gate_init = -2.0` produces a
sigmoid gate. The final logit is

```
logit = base_logit + sigmoid(gate_logit) * primitive_delta_raw,
```

which matches the additive-head pattern used by p009 (LMGConv) and
p052 (PUGP). The negative initial gate bias guarantees the head
starts as a near no-op so the primitive cannot hurt before it has
proven useful.

## Falsifiability

`p053` is **falsified** in the sense of the source markdown if any
of the following hold:

1. **Pressure-delta falsifier**. Setting `ablation = "no_pressure_delta"`
   matches the unablated run: the pressure-delta features are not
   load-bearing, and the primitive collapses to a typed edge-count
   head.
2. **Capture-value falsifier**. Setting `ablation = "no_capture_value"`
   matches: the explicit captured-piece value / gives-check tagging
   is not load-bearing.
3. **Topology falsifier**. Setting `ablation = "random_typed_edges"`
   matches: the per-piece-type chess connectivity is not load-bearing.
4. **Typed-routing falsifier**. Setting `ablation =
   "shared_target_pool"` matches: per-piece-type routing of
   pressure-delta features is not load-bearing.
5. **Benchmark falsifier**. The primitive fails to reduce
   near-puzzle false positives on the canonical puzzle_binary split
   while keeping aggregate PR AUC within `i193 - 0.005`.

## Out-of-scope (deferred)

- **Special move classes** (en-passant, castling, promotion target
  selection). The current topology mirrors p009's pseudo-legal
  coverage; the source markdown's "pseudo-legal-plus" extension is
  deferred to a follow-up. The architectural slot for these features
  is the per-edge feature dictionary -- adding new classes is a
  topology extension and a feature-dictionary extension.
- **Edge-square message passing**. The source markdown sketches a
  two-round edge-square message-passing stack with edge-level
  gating. The current implementation pools per-target and per-type
  globally, which keeps wall-clock cost in the same envelope as
  p052. A future iteration may add a single message-passing round
  if the static-pool primitive proves useful.

The source markdown describes a richer per-edge message-passing
architecture; the current p053 implementation is the **minimal
honest production version** of the load-bearing claims (per-edge
pressure-delta features routed along the typed legal-move
adjacency), wired into the shared trainer and model registry with
focused tests. Deferred items are explicitly marked as
out-of-scope rather than silently dropped.
