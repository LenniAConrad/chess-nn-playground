# Math Thesis

Source: `ideas/research/primitives/external_23_sparse_legal_move_router_kinematic_state_space.md`
(Sparse Legal-Move Router, SLMR — first-ranked proposal).

## Operator

Let `X in R^{B x 64 x D}` be a per-square embedding stack and
`M in {0, 1}^{B x 64 x 64}` the rule-exact legal-move adjacency derived
from the simple_18 board. SLMR computes one round of masked attention:

```
attn_{i, j} = (Q_i . K_j) / sqrt(d_attn)   if M_{i, j} = 1
            = -inf                          otherwise
weights_i   = softmax(attn_i)
y_i         = sum_j weights_{i, j} * V_j
```

Sources `i` with no legal targets (empty squares, pieces with no moves)
fall back to a self-loop so softmax does not NaN; their output
contribution is then masked out so they cannot pollute the pooled
feature.

The "graph" used here is the *aggregated own-piece legal-move adjacency*:
for each own piece on square `i`, `M_{i, j} = 1` whenever the piece could
move to square `j` under standard chess rules (including blocker
termination for sliders, knight L-moves, king step, pawn forward/capture
geometry).

## What is proven

- The adjacency `M` matches the i193 geometry tables exactly: jump pieces
  use the per-(piece-type, color, source, target) attack mask;
  sliding pieces additionally require the in-between line to be clear,
  using the `between` table + occupancy. Both helpers are imported from
  the i193 trunk module.
- The masked softmax is a standard PyTorch construction with the usual
  numerical-stability guarantees.

## What is hypothesised

- Information that flows along legal-move edges is more relevant to
  puzzle classification than information aggregated by an unconstrained
  attention (or a 3x3 conv stack). In particular, fork-style tactics
  hinge on a single piece routing to two distinct legal targets — an
  interaction that the trunk encodes only implicitly.

## Architecture-level claim

```
final_logit(x) = i193_trunk(x) + primitive_gate(x) * primitive_delta(x)
```

The router's pooled feature is concatenated with the trunk diagnostics
to produce both the gate logit and the delta.

## Failure cases

- For positions with few legal moves (king-stripped endgames), the
  routed feature may be all-self-loops with no real information.
- An unconstrained mask (every square attends to every square) may carry
  comparable information on the scout split, in which case the topology
  is not load-bearing.
- The (B, 64, 64) attention matmul dominates the per-step cost; the head
  is more expensive than `p025`/`p028` but cheaper than the trunk.

## Falsifiers

- `full_64x64_mask`: ignore the legal-move adjacency and let attention
  route everywhere. If the unablated and ablated runs match, the
  topology is not load-bearing.
- `self_loop_only`: only allow each square to attend to itself. If this
  matches the unablated run, the head is just doing per-square feature
  mixing without using the graph.
- `shuffle_adjacency`: random permutation of the mask rows and columns.
  Decouples the legal-move pattern from the actual board squares.
- `zero_router_features`: zero the pooled output; tests the trunk
  diagnostics' contribution to the fusion vector.
