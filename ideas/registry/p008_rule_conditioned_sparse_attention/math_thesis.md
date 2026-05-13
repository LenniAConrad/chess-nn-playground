# Mathematical Thesis — p008 Rule-Conditioned Sparse Attention (MobScan)

## Operator signature

Let `X ∈ R^{B × 64 × d}` be per-square token embeddings and let
`A(X_b) ∈ {0, 1}^{64 × 64}` be the input-determined legal-move adjacency
(deterministic discrete function of the simple_18 board). For input-
conditioned gates `A_s, B_s, C_s ∈ R^{state_dim}` (Mamba/S6-style):

```
h^0_s   = B_s ⊙ x_s                                         (input injection)
h^{t+1}_s = A_s ⊙ mean_{p ∈ parents(s)} h^t_p + B_s ⊙ x_s    (selective scan)
y_s      = C_s ⊙ h^T_s                                       (read-out)
```

with `parents(s) = {p : A(X)_{p,s} = 1}` and `T = num_iterations` (3 in
the default config). Aggregation over parents is degree-normalised mean
(GraphSAGE-style) to keep the recurrence numerically stable on
positions with very heterogeneous mobility.

The combined model is the additive, gated head

```
final_logit = base_logit + σ(W_gate · trunk_pool) · MLP(mean_pool(y))
```

## Why this is not just masked attention or a fixed-graph SSM

- Masked attention with `attn_mask = A(X)` still allocates an n^2
  logit matrix and uses softmax; MobScan never instantiates a dense
  score map and propagates information by selective recurrence, not by
  dot-product matching.
- Mamba's hardware-aware `selective_scan_cuda` assumes a fixed 1-D
  causal parent relation; MobScan replaces it with a per-batch
  input-determined DAG indexed by chess legal moves. The eager mode
  unroll over `num_iterations` is the dense-batch analogue of the
  fused scan kernel; the operator's structure (input-conditioned gates
  over rule-derived parents) is preserved.

## Claimed advantage

- Multi-step tactical threats (deep attack chains) require several
  conv hops to propagate; MobScan delivers `num_iterations` legal-move
  hops in one operator call.
- The cost is `O(B · num_iterations · 64^2 · state_dim)` worst case,
  but the adjacency is sparse (`|E| ≈ 30`), so the effective work is
  `O(B · num_iterations · |E| · state_dim)`.

## What is actually proven

- Forward pass shape, gradient flow, and gate value bounds (`A, B, C
  ∈ [0, 1]^{state_dim}`) are unit-tested.
- The recurrence reduces to a deterministic state-zero output when
  `dense_edges` is set with `A_s = 0`.

## What is hypothesised

- The MobScan spec's "match or beat i193 at ≤1.2x wall-clock" target
  is **not** measured. Scout training is gated by `CLAUDE_ALLOW_TRAINING=1`.
- The `random_edges` ablation tests whether the rule-derived graph
  topology is load-bearing; `single_iteration` tests whether the
  multi-hop propagation is load-bearing beyond a single message-pass
  step.

## Failure cases

- Vanishing / exploding products along long mobility chains (e.g.,
  queens on open lines) are the dominant numerical risk; the operator
  uses `A_s = sigmoid(...)`, which contracts in [0, 1] and is exactly
  the Mamba-2 stabilisation strategy.
- Dense recurrence over the 64x64 mask is unavoidable in eager mode;
  a fused custom kernel is the production path for engine deployment.
