# Math Thesis

Source: `ideas/research/primitives/external_30_sparse_legal_graph_transition_delta_accumulator.md`
(Section "primitive_sparse_transition_flow"; first-listed proposal).

## Working thesis

For a position ``x`` with simple_18 board tensor and per-square seed
feature ``X in R^{B x 64 x d}``:

1. Compute the per-board legal-move adjacency ``A in {0, 1}^{64 x 64}``
   via the legal_move_graph helper (blocker-resolved chess-rule
   geometry).
2. Define a learned joint edge function

       phi(X_i, X_j) = LayerNorm(ReLU(
           W_self  X_i
         + W_neighbor X_j
         + W_interact (X_i ⊙ X_j)
       ))

   with ``W_self, W_neighbor, W_interact in R^{d -> d_edge}``. The
   Hadamard interaction term is the key joint non-separable factor:
   removing it (the ``separable_phi`` ablation) reduces the operator to
   ``ReLU(W_self X_i + W_neighbor X_j)``, which is a separable
   sum-of-linears.
3. Aggregate by source square:

       Y[i] = (1 / max(deg(i), 1)) * sum_{j : A[i, j] = 1} phi(X_i, X_j).

   Mean aggregation prevents high-degree positions from saturating the
   delta head.
4. Pool ``Y`` and project to a scalar logit delta gated by trunk
   diagnostics plus the per-sample edge-magnitude summary.

## Why this matters

Standard GAT applies a separable score and softmax-normalised
attention. SLMGT applies a *joint* edge function with a hard binary
chess-rule mask. The Hadamard interaction lets the operator learn
"attacker-defender pair" features: ``X_i ⊙ X_j`` is nonzero only when
both squares carry compatible feature signals. This is the right
inductive bias for hanging-piece / pin / fork detection.

## What is actually proven

The aggregation is well-defined for any board. The Hadamard interaction
makes ``phi`` non-separable through any single Linear; removing it
strictly reduces the operator's capacity (control by the
``separable_phi`` ablation).

## What is only hypothesized

That the joint edge function outperforms the separable phi
(``separable_phi`` ablation) and the rule-graph-removed control
(``uniform_adjacency`` ablation).

## Failure cases

1. *Hidden rebrand of separable GAT*: tested by ``separable_phi``.
2. *Memory-bound at large batch*: the (B, 64, 64, d_edge) pair tensor
   is O(B * 64 * 64 * d) memory. At default sizes this is ~50MB for
   B=128. Larger batches require an explicit sparse-edge formulation.
3. *Adjacency unimportant*: if ``uniform_adjacency`` matches, the
   chess-rule mask carries no signal beyond what the i193 trunk
   already encodes.

## Falsifier

- ``separable_phi`` -- removes the Hadamard interaction. Tests whether
  the joint edge function is load-bearing.
- ``uniform_adjacency`` -- replaces ``A`` with all-ones (minus
  identity). Tests whether the chess-rule mask is load-bearing.
- ``shuffle_adjacency`` -- batch-permutes ``A``. Decouples the rule
  indicators from the position; rule-feature falsifier.
