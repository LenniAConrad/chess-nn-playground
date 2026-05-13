# Mathematical Thesis — p011 Legal-Edge Compile Scatter

## Operator signature

For per-square token embeddings `X ∈ R^{B × 64 × d}` and the per-piece-
type legal-move adjacency `A_r ∈ {0, 1}^{64 × 64}` (derived from the
simple_18 board), define the per-edge σ-gate and message

```
g_{r, i, j} = A_r(X)_{i, j} · σ( a_r · [x_i, x_j] )
m_{r, i, j} = W_r · x_i
```

The per-destination output is a degree-normalised scatter

```
y_j = Σ_r ( 1 / max(eps, Σ_i g_{r, i, j}) ) Σ_i g_{r, i, j} · m_{r, i, j}
```

LayerNorm normalises the typed sum, mean-pooling over destinations
yields a `(B, m)` feature collapsed by the delta MLP.

```
final_logit = base_logit + σ(W_gate · trunk_pool) · MLP(LN(Σ_r y^{(r)}))
```

## Why this is not just typed attention

- The discrete adjacency `A_r` is a deterministic chess-rule function
  of the board with `stop_gradient` — the dense n^2 attention logit
  matrix is never instantiated.
- The σ-gate is computed per edge from the `(source, destination)`
  feature pair, not from a softmax over destinations; multiple edges
  can co-exist with high gate values, which is the file's "overlapping
  hyperedge" argument adapted to typed move edges.
- Per-type `W_r` keep typed information disentangled from p009's
  perspective, but the σ-gate adds a feature-conditioned weighting
  layer that p009 lacks.

## Claimed advantage

- The operator handles attacker / defender / overloaded-piece
  interactions where the same legal move's importance depends on
  contextual features — knight forking a king vs a rook in an
  endgame are both knight legal-edges, but the σ-gate can amplify the
  former and suppress the latter.
- Cost is `O(B · 6 · 64^2 · (2d + edge_gate_hidden))` for the gate
  MLP plus `O(B · 6 · 64^2 · m)` for the bmm; the mask sparsity zeros
  most of the contribution.

## What is actually proven

- Forward pass shape, gradient flow, and gate / message norms behave
  sensibly on unit-tested FENs.
- The `no_edge_gate` ablation degenerates to a typed weighted scatter
  (a stricter version of p009 LMGConv).

## What is hypothesised

- The file's "improve over both i193 and a precomputed-edge GAT
  baseline" target is **not** measured. Scout training is gated by
  `CLAUDE_ALLOW_TRAINING=1`.
- `no_edge_gate` tests whether the σ-gate is load-bearing;
  `random_typed_edges` tests whether the rule structure is load-bearing.

## Failure cases

- The eager-mode dense `(B, 6, 64, 64, 2d)` materialisation is memory-
  intensive for large batches; default config uses `token_embed_dim
  = 32` and `edge_gate_hidden = 16` to keep memory manageable.
- The per-edge gate MLP is unrolled by piece type; a single fused
  per-type-batched MLP would reduce kernel launches.
