# Architecture

`Pareto Antichain Frontier Network` (p001) is an additive, gated head
on top of the existing i193 `ExchangeThenKingDualStreamNetwork` trunk.
The trunk runs once per board; the head adds a candidate-utility
table, the Pareto-antichain operator, and a small fusion MLP.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit for the BCE-with-logits
`puzzle_binary` trainer, plus a rich per-sample diagnostics dict.

## Mechanism

1. **i193 trunk pass.** ``run_i193_trunk_with_spatial`` runs the
   trunk's feature builder and encoders once, returning the base logit,
   the full i193 diagnostic dict, the joint pool feature `p_trunk`,
   and the spatial map `H = cat(ex_h, kg_h)` of shape
   `(B, 2 * channels, 8, 8)`.

2. **Candidate compiler.** A `BoardTokenAttention` pool with
   `K = num_candidates` learnable queries (default 16) and
   `D = token_dim` (default 48) attends over the 64 spatial cells of
   `H` to produce candidate tokens `tokens_k in R^{B x K x D}` and
   per-square attention weights `alpha_k in R^{B x K x 64}`.

3. **Utility table.** A learned bilinear projection conditions each
   candidate token on the trunk pool feature:

   ```
   ctx = context_proj(p_trunk)             # (B, D)
   gated_tokens = tokens_k * tanh(ctx)     # (B, K, D)
   U = utility_head(gated_tokens)          # (B, K, C)
   ```

   with `C = utility_channels` (default 6). Larger `U_{kc}` means
   higher utility on channel `c`. Channels are learned ends, not
   fixed to specific tactical motifs.

4. **PAFR operator.** ``pareto_antichain_frontier`` computes the
   pairwise dominance product, log-domain non-dominated probability,
   frontier softmax with mixing factor `beta` and temperature
   `tau_set`, frontier-weighted summary, width, and entropy. See
   `math_thesis.md` for the math.

5. **Fusion head.** The summary `summary in R^{B x D}`, width
   `width in R^{B}`, entropy `entropy in R^{B}`, and trunk pool
   feature `p_trunk` are concatenated and fed into a small LayerNorm
   + GELU MLP that produces `primitive_delta_raw`. A second MLP over
   `[p_trunk, width, entropy]` produces the gate logit. Final
   contract:

   ```
   primitive_delta = primitive_gate * primitive_delta_raw
   final_logit     = base_logit + primitive_delta
   ```

   The gate bias is initialised at `-2.0` so the primitive starts as
   a near-no-op and the trunk's baseline behaviour is preserved early
   in training.

6. **Ablations.** Eight supported modes via `model.ablation`:

   - `none`: full architecture (default).
   - `scalar_max`: collapse utilities to per-candidate scalar max
     before reducing — collapses the partial order to a total order.
   - `single_channel`: use only utility channel 0 — collapses the
     product partial order to a 1-D partial order.
   - `shuffle_channels`: permute utility channels across candidates
     in-batch — decouples channels from candidates while keeping
     marginal channel distributions.
   - `uniform_frontier`: drop the frontier softmax to a uniform
     distribution over valid candidates.
   - `disable_gate`: hold the gate at 1.0.
   - `zero_delta`: hold `primitive_delta` at 0 — recovers the i193
     baseline.
   - `trunk_only`: zero out both features and delta — strongest
     control.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed by the model. The candidate tokens are compiled
inside the model from the trunk's spatial features only, which are in
turn computed from the `simple_18` board tensor.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| BoardTokenAttention | `O(K * HW * D)` set-query attention pool |
| Utility table | `O(K * D * head_hidden_dim + K * C * head_hidden_dim)` |
| PAFR operator | `O(K^2 * C)` pairwise dominance product |
| Fusion MLPs | `O((D + 2 + joint_dim) * head_hidden_dim)` |

For `K = 16` and `C = 6` the dominance product is `O(B * 1536)`, far
cheaper than the trunk forward.

## Implementation Binding

- Registered model name: `pareto_antichain_frontier_network`.
- Source implementation:
  `src/chess_nn_playground/models/primitives/pareto_antichain_frontier_network.py`.
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Trunk source:
  `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`
  (the bespoke i193 trunk is wrapped, not reimplemented).
- Idea-local wrapper:
  `ideas/registry/p001_pareto_antichain_frontier/model.py`.
- Training config:
  `ideas/registry/p001_pareto_antichain_frontier/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/registry.py`:
  `MODEL_BUILDERS["pareto_antichain_frontier_network"] = build_pareto_antichain_frontier_network_from_config`.
